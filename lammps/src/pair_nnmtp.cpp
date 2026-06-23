/* ----------------------------------------------------------------------
   LAMMPS pair_style nnmtp
   Pure C++ implementation of NN-MTP with ANALYTICAL gradients
------------------------------------------------------------------------- */

#include "pair_nnmtp.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>

using namespace LAMMPS_NS;

PairNNMTP::PairNNMTP(LAMMPS *lmp) : Pair(lmp)
{
  n_species = 0; n_radial_basis = 0; n_radial_funcs = 0;
  descriptor_dim = 0; embed_dim = 0; r_min = 0.0; r_max = 0.0;
  single_enable = 0; restartinfo = 0; manybody_flag = 1;
  use_zbl = true; zbl_r_inner = 0.5; zbl_r_outer = 2.0;
  for (int i = 0; i < 4; i++) species_Z[i] = 0;  // populated by load_model
}

/* ---------------------------------------------------------------------- */
// Periodic table lookup: element symbol -> atomic number
static int element_name_to_Z(const std::string &name) {
  static const std::vector<std::pair<std::string,int>> table = {
    {"H",1},{"He",2},{"Li",3},{"Be",4},{"B",5},{"C",6},{"N",7},{"O",8},
    {"F",9},{"Ne",10},{"Na",11},{"Mg",12},{"Al",13},{"Si",14},{"P",15},
    {"S",16},{"Cl",17},{"Ar",18},{"K",19},{"Ca",20},{"Sc",21},{"Ti",22},
    {"V",23},{"Cr",24},{"Mn",25},{"Fe",26},{"Co",27},{"Ni",28},{"Cu",29},
    {"Zn",30},{"Ga",31},{"Ge",32},{"As",33},{"Se",34},{"Br",35},{"Kr",36},
    {"Rb",37},{"Sr",38},{"Y",39},{"Zr",40},{"Nb",41},{"Mo",42},{"Tc",43},
    {"Ru",44},{"Rh",45},{"Pd",46},{"Ag",47},{"Cd",48},{"In",49},{"Sn",50},
    {"Sb",51},{"Te",52},{"I",53},{"Xe",54},{"Cs",55},{"Ba",56},{"La",57},
    {"Ce",58},{"Pr",59},{"Nd",60},{"Pm",61},{"Sm",62},{"Eu",63},{"Gd",64},
    {"Tb",65},{"Dy",66},{"Ho",67},{"Er",68},{"Tm",69},{"Yb",70},{"Lu",71},
    {"Hf",72},{"Ta",73},{"W",74},{"Re",75},{"Os",76},{"Ir",77},{"Pt",78},
    {"Au",79},{"Hg",80},{"Tl",81},{"Pb",82},{"Bi",83},{"Po",84},{"At",85},
    {"Rn",86},{"Fr",87},{"Ra",88},{"Ac",89},{"Th",90},{"Pa",91},{"U",92},
  };
  for (auto &kv : table) if (kv.first == name) return kv.second;
  return -1;
}

PairNNMTP::~PairNNMTP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairNNMTP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairNNMTP::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style nnmtp command");
}

void PairNNMTP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg < 3) error->all(FLERR, "Incorrect pair_coeff command");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for nnmtp must be * *");

  load_model(std::string(arg[2]));

  int ntypes = atom->ntypes;
  type_map.resize(ntypes + 1, -1);

  if (narg - 3 != ntypes)
    error->all(FLERR, "Number of element names must match atom types");

  // Map each user-supplied element name to its position in species_Z (which was
  // read from the binary). This way the binary encodes the training species
  // order, and the user provides any element names in any order.
  for (int i = 0; i < ntypes; i++) {
    std::string name = arg[3 + i];
    int Z = element_name_to_Z(name);
    if (Z < 0)
      error->all(FLERR, ("Unknown element symbol: " + name).c_str());
    for (int j = 0; j < n_species; j++) {
      if (species_Z[j] == Z) { type_map[i + 1] = j; break; }
    }
    if (type_map[i + 1] < 0)
      error->all(FLERR, ("Element " + name +
                         " not in trained species_list").c_str());
  }

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

void PairNNMTP::init_style()
{
  // Full list needed because descriptor depends on ALL neighbors
  // BUT: only use full list for descriptor computation
  // Force application handles Newton 3rd + ghost mapping manually
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairNNMTP::init_one(int i, int j) { return r_max; }

/* ---------------------------------------------------------------------- */

void PairNNMTP::load_model(const std::string &filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    error->all(FLERR, ("Cannot open: " + filename).c_str());

  char magic[6]; file.read(magic, 6);
  if (std::string(magic, 6) != "NNMTP1")
    error->all(FLERR, "Invalid model file");

  file.read((char *)&n_species, sizeof(int));
  if (n_species > 4)
    error->all(FLERR, "species_Z fixed-size array supports up to 4 species");
  for (int i = 0; i < n_species; i++)
    file.read((char *)&species_Z[i], sizeof(int));
  file.read((char *)&n_radial_basis, sizeof(int));
  file.read((char *)&n_radial_funcs, sizeof(int));
  file.read((char *)&r_min, sizeof(double));
  file.read((char *)&r_max, sizeof(double));
  file.read((char *)&descriptor_dim, sizeof(int));
  file.read((char *)&embed_dim, sizeof(int));

  // ZBL section: flag + (r_inner, r_outer) if flag==1.
  int zbl_flag = 0;
  file.read((char *)&zbl_flag, sizeof(int));
  if (zbl_flag) {
    use_zbl = true;
    double ri_d, ro_d;
    file.read((char *)&ri_d, sizeof(double));
    file.read((char *)&ro_d, sizeof(double));
    zbl_r_inner = ri_d;
    zbl_r_outer = ro_d;
  } else {
    use_zbl = false;
  }

  int n_mlp_layers;
  file.read((char *)&n_mlp_layers, sizeof(int));
  mlp_dims.resize(n_mlp_layers + 1);
  for (int i = 0; i <= n_mlp_layers; i++)
    file.read((char *)&mlp_dims[i], sizeof(int));

  auto read_vec = [&](std::vector<double> &v) {
    int n; file.read((char *)&n, sizeof(int));
    v.resize(n);
    for (int i = 0; i < n; i++) file.read((char *)&v[i], sizeof(double));
  };

  read_vec(radial_coeffs);
  read_vec(species_embed);
  read_vec(energy_shift);

  int n_params; file.read((char *)&n_params, sizeof(int));
  mlp_weights.resize(n_params);
  for (int i = 0; i < n_params; i++) read_vec(mlp_weights[i]);

  file.close();

  if (comm->me == 0) {
    printf("NN-MTP C++ loaded: %d species, desc=%d, MLP:", n_species, descriptor_dim);
    for (int i = 0; i <= n_mlp_layers; i++) printf(" %d", mlp_dims[i]);
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

inline double PairNNMTP::cutoff_fn(double r)
{
  if (r >= r_max) return 0.0;
  double x = r / r_max;
  double t = 1.0 - x * x;
  return t * t;
}

inline double PairNNMTP::silu(double x)
{
  double s = 1.0 / (1.0 + exp(-x));
  return x * s;
}

void PairNNMTP::zbl_pair(double r, int Z_i, int Z_j, double &e_zbl, double &de_zbl_dr)
{
  // ZBL potential with smooth switching
  double Zi = (double)Z_i, Zj = (double)Z_j;
  double a = 0.4543 * 0.529 / (pow(Zi, 0.23) + pow(Zj, 0.23));
  double x = r / a;

  double c[4] = {0.1818, 0.5099, 0.2802, 0.02817};
  double d[4] = {-3.2, -0.9423, -0.4028, -0.2016};

  double phi = 0.0, dphi_dx = 0.0;
  for (int k = 0; k < 4; k++) {
    double ex = exp(d[k] * x);
    phi += c[k] * ex;
    dphi_dx += c[k] * d[k] * ex;
  }
  double dphi_dr = dphi_dx / a;

  double coulomb = 14.3996 * Zi * Zj;
  double r_safe = (r < 1e-6) ? 1e-6 : r;
  double e_raw = coulomb / r_safe * phi;
  double de_raw = coulomb * (-phi / (r_safe * r_safe) + dphi_dr / r_safe);

  // Switching function: (1-t)^3 * (1+3t+6t^2), t = (r-r_inner)/(r_outer-r_inner)
  double sw, dsw_dr;
  if (r <= zbl_r_inner) {
    sw = 1.0; dsw_dr = 0.0;
  } else if (r >= zbl_r_outer) {
    sw = 0.0; dsw_dr = 0.0;
  } else {
    double t = (r - zbl_r_inner) / (zbl_r_outer - zbl_r_inner);
    double omt = 1.0 - t;
    sw = omt * omt * omt * (1.0 + 3.0*t + 6.0*t*t);
    double dsw_dt = -3.0*omt*omt*(1.0+3.0*t+6.0*t*t) + omt*omt*omt*(3.0+12.0*t);
    dsw_dr = dsw_dt / (zbl_r_outer - zbl_r_inner);
  }

  e_zbl = e_raw * sw;
  de_zbl_dr = de_raw * sw + e_raw * dsw_dr;
}

void PairNNMTP::chebyshev_basis(double r, double *basis, int n)
{
  double x = 2.0 * (r - r_min) / (r_max - r_min) - 1.0;
  if (x < -1.0) x = -1.0;
  if (x > 1.0) x = 1.0;
  basis[0] = 1.0;
  if (n > 1) basis[1] = x;
  for (int i = 2; i < n; i++)
    basis[i] = 2.0 * x * basis[i - 1] - basis[i - 2];
}

/* ---------------------------------------------------------------------- */

void PairNNMTP::compute_descriptors(int i, int itype,
                                     int *jlist, int jnum, double *desc)
{
  double **x = atom->x;
  int *type = atom->type;
  int n_mu = n_radial_funcs;
  int n_pairs = n_species * n_species;

  std::vector<double> M0(n_mu, 0.0);
  std::vector<double> M1(n_mu * 3, 0.0);
  std::vector<double> M2(n_mu * 9, 0.0);
  double basis[32];

  for (int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj] & NEIGHMASK;
    int jtype = type_map[type[j]];
    if (jtype < 0) continue;

    double dx = x[j][0] - x[i][0];
    double dy = x[j][1] - x[i][1];
    double dz = x[j][2] - x[i][2];
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    if (r >= r_max || r < 1e-8) continue;

    double fc = cutoff_fn(r);
    double ri = 1.0 / r;
    double ux = dx*ri, uy = dy*ri, uz = dz*ri;

    chebyshev_basis(r, basis, n_radial_basis);
    for (int b = 0; b < n_radial_basis; b++) basis[b] *= fc;

    int pair_idx = itype * n_species + jtype;
    for (int mu = 0; mu < n_mu; mu++) {
      double fm = 0.0;
      for (int b = 0; b < n_radial_basis; b++)
        fm += radial_coeffs[mu*n_pairs*n_radial_basis + pair_idx*n_radial_basis + b] * basis[b];

      M0[mu] += fm;
      M1[mu*3+0] += fm*ux; M1[mu*3+1] += fm*uy; M1[mu*3+2] += fm*uz;
      M2[mu*9+0] += fm*ux*ux; M2[mu*9+4] += fm*uy*uy; M2[mu*9+8] += fm*uz*uz;
      M2[mu*9+1] += fm*ux*uy; M2[mu*9+2] += fm*ux*uz; M2[mu*9+5] += fm*uy*uz;
      M2[mu*9+3] += fm*uy*ux; M2[mu*9+6] += fm*uz*ux; M2[mu*9+7] += fm*uz*uy;
    }
  }

  int idx = 0;
  for (int mu = 0; mu < n_mu; mu++) desc[idx++] = M0[mu];
  for (int mu = 0; mu < n_mu; mu++) {
    double sq = 0; for (int d=0;d<3;d++) sq += M1[mu*3+d]*M1[mu*3+d]; desc[idx++] = sq;
  }
  for (int mu1=0; mu1<n_mu; mu1++)
    for (int mu2=mu1+1; mu2<n_mu; mu2++) {
      double dot=0; for(int d=0;d<3;d++) dot+=M1[mu1*3+d]*M1[mu2*3+d]; desc[idx++]=dot;
    }
  for (int mu=0; mu<n_mu; mu++)
    desc[idx++] = M2[mu*9+0]+M2[mu*9+4]+M2[mu*9+8];
}

/* ---------------------------------------------------------------------- */

// MLP forward with intermediate activations saved for backprop
double PairNNMTP::mlp_forward(double *input, int input_dim)
{
  std::vector<double> current(input, input + input_dim);
  int n_layers = (int)mlp_weights.size() / 2;

  for (int layer = 0; layer < n_layers; layer++) {
    const auto &W = mlp_weights[layer*2];
    const auto &b = mlp_weights[layer*2+1];
    int out_dim = (int)b.size();
    int in_dim = (int)current.size();
    std::vector<double> next(out_dim);
    for (int o = 0; o < out_dim; o++) {
      double sum = b[o];
      for (int i = 0; i < in_dim; i++) sum += W[o*in_dim+i]*current[i];
      if (layer < n_layers-1) sum = silu(sum);
      next[o] = sum;
    }
    current = next;
  }
  return current[0];
}

// MLP forward + backward: returns dE/d_input
void mlp_forward_backward(const std::vector<std::vector<double>> &weights,
                          double *input, int input_dim,
                          double *grad_input, double &energy)
{
  int n_layers = (int)weights.size() / 2;

  // Forward pass - save pre-activations and activations
  std::vector<std::vector<double>> pre_acts(n_layers);
  std::vector<std::vector<double>> acts(n_layers + 1);
  acts[0].assign(input, input + input_dim);

  for (int layer = 0; layer < n_layers; layer++) {
    const auto &W = weights[layer*2];
    const auto &b = weights[layer*2+1];
    int out_dim = (int)b.size();
    int in_dim = (int)acts[layer].size();
    pre_acts[layer].resize(out_dim);
    acts[layer+1].resize(out_dim);

    for (int o = 0; o < out_dim; o++) {
      double sum = b[o];
      for (int i = 0; i < in_dim; i++) sum += W[o*in_dim+i]*acts[layer][i];
      pre_acts[layer][o] = sum;
      if (layer < n_layers-1) {
        double sig = 1.0/(1.0+exp(-sum));
        acts[layer+1][o] = sum * sig;  // SiLU
      } else {
        acts[layer+1][o] = sum;  // linear
      }
    }
  }

  energy = acts[n_layers][0];

  // Backward pass
  // dL/d_output = 1.0
  std::vector<double> grad_out = {1.0};

  for (int layer = n_layers-1; layer >= 0; layer--) {
    const auto &W = weights[layer*2];
    const auto &b = weights[layer*2+1];
    int out_dim = (int)b.size();
    int in_dim = (int)acts[layer].size();

    // Apply activation gradient (SiLU' = sigmoid + x*sigmoid*(1-sigmoid))
    std::vector<double> grad_pre(out_dim);
    for (int o = 0; o < out_dim; o++) {
      if (layer < n_layers-1) {
        double z = pre_acts[layer][o];
        double sig = 1.0/(1.0+exp(-z));
        double dsilu = sig + z * sig * (1.0 - sig);
        grad_pre[o] = grad_out[o] * dsilu;
      } else {
        grad_pre[o] = grad_out[o];
      }
    }

    // Gradient w.r.t. input of this layer
    std::vector<double> grad_in(in_dim, 0.0);
    for (int i = 0; i < in_dim; i++)
      for (int o = 0; o < out_dim; o++)
        grad_in[i] += W[o*in_dim+i] * grad_pre[o];

    grad_out = grad_in;
  }

  // Copy gradient to output
  for (int i = 0; i < input_dim; i++)
    grad_input[i] = grad_out[i];
}

/* ---------------------------------------------------------------------- */

void PairNNMTP::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int n_mu = n_radial_funcs;
  int n_pairs = n_species * n_species;
  int total_input = descriptor_dim + embed_dim;

  double total_eng = 0.0;

  // Per-atom pair force buffer for thread-safe accumulation
  struct PairForce { int i, j; double fx, fy, fz; };
  std::vector<std::vector<PairForce>> all_pair_forces(inum);
  std::vector<double> atom_energies(inum, 0.0);

  #pragma omp parallel for reduction(+:total_eng) schedule(dynamic)
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int itype = type_map[type[i]];
    if (itype < 0) continue;

    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    // Thread-local variables
    std::vector<double> desc(descriptor_dim);
    std::vector<double> full_input(total_input);
    std::vector<double> grad_input(total_input);
    std::vector<double> M0(n_mu, 0.0);
    std::vector<double> M1(n_mu * 3, 0.0);
    std::vector<double> M2(n_mu * 9, 0.0);

    double basis[32], dbasis_dr[32];

    // Per-neighbor data for gradient
    struct NeighData {
      int j, jtype;
      double dx, dy, dz, r;
      double f_mu[16];       // radial functions
      double df_mu_dr[16];   // d(f_mu)/dr
    };
    std::vector<NeighData> neigh_data;

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      int jtype = type_map[type[j]];
      if (jtype < 0) continue;

      double dx = x[j][0]-x[i][0], dy = x[j][1]-x[i][1], dz = x[j][2]-x[i][2];
      double r = sqrt(dx*dx+dy*dy+dz*dz);
      if (r >= r_max || r < 1e-8) continue;

      double fc = cutoff_fn(r);
      double ri = 1.0/r;
      double ux=dx*ri, uy=dy*ri, uz=dz*ri;

      // Cutoff derivative: d(fc)/dr
      double xr = r / r_max;
      double dfc_dr = -4.0 * xr * (1.0 - xr*xr) / r_max;

      // Chebyshev basis and derivative
      chebyshev_basis(r, basis, n_radial_basis);

      // Chebyshev derivative: dT_n/dx * dx/dr
      // When r < r_min or r > r_max, x is clamped → gradient must be 0
      double dx_dr = (r >= r_min && r <= r_max) ? 2.0 / (r_max - r_min) : 0.0;
      double xx = 2.0*(r-r_min)/(r_max-r_min) - 1.0;
      if (xx < -1.0) xx = -1.0; if (xx > 1.0) xx = 1.0;

      dbasis_dr[0] = 0.0;  // dT_0/dr = 0
      if (n_radial_basis > 1) dbasis_dr[1] = dx_dr;  // dT_1/dr = dx/dr
      for (int b = 2; b < n_radial_basis; b++)
        dbasis_dr[b] = 2.0*(basis[b-1]*dx_dr + xx*dbasis_dr[b-1]) - dbasis_dr[b-2];

      // basis_with_cutoff = basis * fc
      // d(basis*fc)/dr = dbasis*fc + basis*dfc
      NeighData nd;
      nd.j = j; nd.jtype = jtype;
      nd.dx = dx; nd.dy = dy; nd.dz = dz; nd.r = r;

      int pair_idx = itype * n_species + jtype;
      for (int mu = 0; mu < n_mu; mu++) {
        double fm = 0.0, dfm = 0.0;
        for (int b = 0; b < n_radial_basis; b++) {
          int cidx = mu*n_pairs*n_radial_basis + pair_idx*n_radial_basis + b;
          double c = radial_coeffs[cidx];
          fm += c * basis[b] * fc;
          dfm += c * (dbasis_dr[b]*fc + basis[b]*dfc_dr);
        }
        nd.f_mu[mu] = fm;
        nd.df_mu_dr[mu] = dfm;

        M0[mu] += fm;
        M1[mu*3+0] += fm*ux; M1[mu*3+1] += fm*uy; M1[mu*3+2] += fm*uz;
        M2[mu*9+0] += fm*ux*ux; M2[mu*9+4] += fm*uy*uy; M2[mu*9+8] += fm*uz*uz;
        M2[mu*9+1] += fm*ux*uy; M2[mu*9+2] += fm*ux*uz; M2[mu*9+5] += fm*uy*uz;
        M2[mu*9+3] += fm*uy*ux; M2[mu*9+6] += fm*uz*ux; M2[mu*9+7] += fm*uz*uy;
      }
      neigh_data.push_back(nd);
    }

    // Build descriptor
    int idx = 0;
    for (int mu=0; mu<n_mu; mu++) desc[idx++] = M0[mu];
    for (int mu=0; mu<n_mu; mu++) {
      double sq=0; for(int d=0;d<3;d++) sq+=M1[mu*3+d]*M1[mu*3+d]; desc[idx++]=sq;
    }
    for (int mu1=0;mu1<n_mu;mu1++)
      for(int mu2=mu1+1;mu2<n_mu;mu2++) {
        double dot=0; for(int d=0;d<3;d++) dot+=M1[mu1*3+d]*M1[mu2*3+d]; desc[idx++]=dot;
      }
    for (int mu=0;mu<n_mu;mu++)
      desc[idx++] = M2[mu*9+0]+M2[mu*9+4]+M2[mu*9+8];

    // Build full input
    for (int d=0; d<descriptor_dim; d++) full_input[d] = desc[d];
    for (int d=0; d<embed_dim; d++)
      full_input[descriptor_dim+d] = species_embed[itype*embed_dim+d];

    // Debug: print atom 0
    if (ii == 0 && update->ntimestep == 0) {
      printf("C++ Atom 0 type: %d, neighbors: %d\n", itype, (int)neigh_data.size());
    }

    // === MLP forward + backward ===
    double e_i;
    mlp_forward_backward(mlp_weights, full_input.data(), total_input,
                         grad_input.data(), e_i);
    e_i += energy_shift[itype];

    // === ZBL energy (force applied together with NN forces below) ===
    double e_zbl_total = 0.0;
    if (use_zbl) {
      int Zi = species_Z[itype];
      for (auto &nd : neigh_data) {
        if (nd.r >= zbl_r_outer) continue;
        int Zj = species_Z[nd.jtype];
        double e_zbl, de_zbl_dr;
        zbl_pair(nd.r, Zi, Zj, e_zbl, de_zbl_dr);
        e_zbl_total += 0.5 * e_zbl;
      }
    }
    e_i += e_zbl_total;

    if (ii == 0 && update->ntimestep == 0) {
      printf("C++ Atom 0 E=%.6f (ZBL=%.6f)\n", e_i, e_zbl_total);
    }

    // === Compute forces via chain rule ===
    // grad_input[0..descriptor_dim-1] = dE/d_desc
    // Need: dE/dr_j = sum_k (dE/d_desc_k) * (d_desc_k / d_r_j)

    // Extract dE/d_desc
    double *dE_dD = grad_input.data();  // [descriptor_dim]

    // For each neighbor, compute force contribution
    for (auto &nd : neigh_data) {
      double r = nd.r;
      double ri = 1.0/r;
      double ri2 = ri*ri;
      double ux = nd.dx*ri, uy = nd.dy*ri, uz = nd.dz*ri;

      double fx_j = 0.0, fy_j = 0.0, fz_j = 0.0;

      // Precompute dM1[mu,a]/dr_j[dim] for ALL mu
      double u[3] = {ux, uy, uz};
      // dM1_all[mu][a][dim]
      double dM1_all[16][3][3];
      for (int mu = 0; mu < n_mu; mu++) {
        double fm = nd.f_mu[mu];
        double dfm = nd.df_mu_dr[mu];
        for (int dim = 0; dim < 3; dim++) {
          for (int a = 0; a < 3; a++) {
            double delta_ad = (a == dim) ? 1.0 : 0.0;
            dM1_all[mu][a][dim] = dfm * u[dim] * u[a]
                                 + fm * (delta_ad - u[a]*u[dim]) * ri;
          }
        }
      }

      // --- nu=0: dE * dM0/dr_j ---
      for (int mu = 0; mu < n_mu; mu++) {
        double dfm = nd.df_mu_dr[mu];
        double dE_dM0 = dE_dD[mu];
        fx_j += dE_dM0 * dfm * ux;
        fy_j += dE_dM0 * dfm * uy;
        fz_j += dE_dM0 * dfm * uz;
      }

      // --- nu=1 self: dE * d(|M1[mu]|^2)/dr_j ---
      for (int mu = 0; mu < n_mu; mu++) {
        int d_idx = n_mu + mu;
        double dE_dM1sq = dE_dD[d_idx];
        for (int dim = 0; dim < 3; dim++) {
          double grad_dim = 0.0;
          for (int a = 0; a < 3; a++)
            grad_dim += 2.0 * M1[mu*3+a] * dM1_all[mu][a][dim];
          if (dim == 0) fx_j += dE_dM1sq * grad_dim;
          if (dim == 1) fy_j += dE_dM1sq * grad_dim;
          if (dim == 2) fz_j += dE_dM1sq * grad_dim;
        }
      }

      // --- nu=1 cross: dE * d(M1[mu1]·M1[mu2])/dr_j ---
      // d(M1[mu1]·M1[mu2])/dr_j = sum_a (M1[mu2,a]*dM1[mu1,a]/dr_j + M1[mu1,a]*dM1[mu2,a]/dr_j)
      {
        int cross_idx = 0;
        int cross_base = 2 * n_mu;
        for (int mu1 = 0; mu1 < n_mu; mu1++) {
          for (int mu2 = mu1+1; mu2 < n_mu; mu2++) {
            double dE_dcross = dE_dD[cross_base + cross_idx];
            for (int dim = 0; dim < 3; dim++) {
              double grad_dim = 0.0;
              for (int a = 0; a < 3; a++) {
                grad_dim += M1[mu2*3+a] * dM1_all[mu1][a][dim];
                grad_dim += M1[mu1*3+a] * dM1_all[mu2][a][dim];
              }
              if (dim == 0) fx_j += dE_dcross * grad_dim;
              if (dim == 1) fy_j += dE_dcross * grad_dim;
              if (dim == 2) fz_j += dE_dcross * grad_dim;
            }
            cross_idx++;
          }
        }
      }

      // --- dD(nu=2, Tr(M2))/dr_j ---
      int trace_base = 2*n_mu + n_mu*(n_mu-1)/2;
      for (int mu = 0; mu < n_mu; mu++) {
        double fm = nd.f_mu[mu];
        double dfm = nd.df_mu_dr[mu];
        double ri_val = 1.0/nd.r;
        double u[3] = {nd.dx*ri_val, nd.dy*ri_val, nd.dz*ri_val};

        int d_idx = trace_base + mu;
        double dE_dTr = dE_dD[d_idx];

        // Tr(M2[mu]) = sum_a M2[mu,a,a] = sum_j f_mu * u_a^2
        // d(Tr)/dr_j[dim] = sum_a d(f_mu*u_a^2)/dr_j[dim]
        for (int dim = 0; dim < 3; dim++) {
          double grad_dim = 0.0;
          for (int a = 0; a < 3; a++) {
            double delta_ad = (a == dim) ? 1.0 : 0.0;
            // d(fm*u_a^2)/d(r_j[dim])
            // = dfm*u_dim*u_a^2 + fm*2*u_a*(delta_ad - u_a*u_dim)/r
            grad_dim += dfm*u[dim]*u[a]*u[a]
                       + fm*2.0*u[a]*(delta_ad - u[a]*u[dim])*ri_val;
          }
          if (dim==0) fx_j += dE_dTr * grad_dim;
          if (dim==1) fy_j += dE_dTr * grad_dim;
          if (dim==2) fz_j += dE_dTr * grad_dim;
        }
      }

      // Add ZBL force to NN force. pf.fx represents dE_i/dx_j, and E_i gets
      // a contribution of 0.5*ZBL(r_ij) from this neighbor. Therefore
      // d(0.5*ZBL)/dx_j = 0.5 * de_zbl_dr * (x_j-x_i)/r. (Previous sign was
      // inverted, which made ZBL attractive instead of repulsive — harmless
      // when all pair distances lie outside r_outer, catastrophic for H-O.)
      if (use_zbl && nd.r < zbl_r_outer) {
        int Zi = species_Z[itype];
        int Zj = species_Z[nd.jtype];
        double e_zbl, de_zbl_dr;
        zbl_pair(nd.r, Zi, Zj, e_zbl, de_zbl_dr);
        double ri_zbl = 1.0 / nd.r;
        double fzbl = 0.5 * de_zbl_dr * ri_zbl;
        fx_j += fzbl * nd.dx;
        fy_j += fzbl * nd.dy;
        fz_j += fzbl * nd.dz;
      }

      // Store force contribution in buffer (no write to f[] here)
      all_pair_forces[ii].push_back({i, nd.j, fx_j, fy_j, fz_j});
    }

    atom_energies[ii] = e_i;
    total_eng += e_i;
  }

  // === Sequential: apply all forces from buffer ===
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if (eflag_atom) eatom[i] += atom_energies[ii];

    for (auto &pf : all_pair_forces[ii]) {
      // Self-force on atom i
      f[pf.i][0] += pf.fx;
      f[pf.i][1] += pf.fy;
      f[pf.i][2] += pf.fz;

      // Newton 3rd on neighbor j
      int real_j = pf.j;
      if (pf.j >= nlocal)
        real_j = atom->map(atom->tag[pf.j]);
      if (real_j >= 0 && real_j < nlocal) {
        f[real_j][0] -= pf.fx;
        f[real_j][1] -= pf.fy;
        f[real_j][2] -= pf.fz;
      }
    }
  }

  if (eflag_global) eng_vdwl += total_eng;

  // Reverse communication: accumulate ghost forces onto local atoms
  comm_reverse = 3;  // 3 values per atom (fx, fy, fz)
  comm->reverse_comm(this);

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

int PairNNMTP::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  double **f = atom->f;
  for (int i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairNNMTP::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  double **f = atom->f;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }
}
