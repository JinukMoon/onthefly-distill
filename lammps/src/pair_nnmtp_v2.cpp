/* ----------------------------------------------------------------------
   LAMMPS pair_style nnmtp2
   Pure C++ implementation of NN-MTP v2 with ANALYTICAL gradients

   v2: 27 basis (LightPFP Table S1), ZBL, no r_min, atomic number based
------------------------------------------------------------------------- */

#include "pair_nnmtp_v2.h"
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

/* ---------------------------------------------------------------------- */

PairNNMTPv2::PairNNMTPv2(LAMMPS *lmp) : Pair(lmp)
{
  n_species = 0; n_radial_basis = 0; n_radial_funcs = 7;
  descriptor_dim = 27; embed_dim = 16; r_max = 6.0;
  use_zbl = true; zbl_r_inner = 0.5; zbl_r_outer = 2.0;
  single_enable = 0; restartinfo = 0; manybody_flag = 1;
}

PairNNMTPv2::~PairNNMTPv2()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairNNMTPv2::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

void PairNNMTPv2::settings(int narg, char ** /*arg*/)
{
  if (narg != 0) error->all(FLERR, "pair_style nnmtp2 takes no arguments");
}

void PairNNMTPv2::coeff(int narg, char **arg)
{
  if (narg < 3 + atom->ntypes)
    error->all(FLERR, "pair_coeff: * * model.bin elem1 elem2 ...");

  if (!allocated) allocate();

  std::string model_file = arg[2];
  load_model(model_file);

  // Map LAMMPS types to species indices via element names
  // arg[3], arg[4], ... are element names for LAMMPS types 1, 2, ...
  type_map.resize(atom->ntypes + 1, -1);

  // Map element name to atomic number
  auto elem_to_z = [](const char *s) -> int {
    if (!strcmp(s,"H")) return 1; if (!strcmp(s,"He")) return 2;
    if (!strcmp(s,"Li")) return 3; if (!strcmp(s,"Be")) return 4;
    if (!strcmp(s,"B")) return 5; if (!strcmp(s,"C")) return 6;
    if (!strcmp(s,"N")) return 7; if (!strcmp(s,"O")) return 8;
    if (!strcmp(s,"F")) return 9; if (!strcmp(s,"Ne")) return 10;
    if (!strcmp(s,"Na")) return 11; if (!strcmp(s,"Mg")) return 12;
    if (!strcmp(s,"Al")) return 13; if (!strcmp(s,"Si")) return 14;
    if (!strcmp(s,"P")) return 15; if (!strcmp(s,"S")) return 16;
    if (!strcmp(s,"Cl")) return 17; if (!strcmp(s,"Ar")) return 18;
    if (!strcmp(s,"K")) return 19; if (!strcmp(s,"Ca")) return 20;
    if (!strcmp(s,"Sc")) return 21; if (!strcmp(s,"Ti")) return 22;
    if (!strcmp(s,"V")) return 23; if (!strcmp(s,"Cr")) return 24;
    if (!strcmp(s,"Mn")) return 25; if (!strcmp(s,"Fe")) return 26;
    if (!strcmp(s,"Co")) return 27; if (!strcmp(s,"Ni")) return 28;
    if (!strcmp(s,"Cu")) return 29; if (!strcmp(s,"Zn")) return 30;
    if (!strcmp(s,"Ga")) return 31; if (!strcmp(s,"Ge")) return 32;
    if (!strcmp(s,"As")) return 33; if (!strcmp(s,"Se")) return 34;
    if (!strcmp(s,"Br")) return 35; if (!strcmp(s,"Kr")) return 36;
    if (!strcmp(s,"Pt")) return 78; if (!strcmp(s,"Au")) return 79;
    return -1;
  };

  for (int i = 1; i <= atom->ntypes; i++) {
    int z = elem_to_z(arg[2 + i]);
    // Find species index
    for (int si = 0; si < n_species; si++) {
      if (species_list[si] == z) {
        type_map[i] = si;
        break;
      }
    }
    if (type_map[i] < 0)
      error->all(FLERR, "pair_coeff: element not in model species_list");
  }

  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      setflag[i][j] = 1;
}

void PairNNMTPv2::init_style()
{
  neighbor->add_request(this, NeighConst::REQ_FULL);
  comm_reverse = 3;
}

double PairNNMTPv2::init_one(int /*i*/, int /*j*/)
{
  return r_max + 0.5;
}

/* ---------------------------------------------------------------------- */

void PairNNMTPv2::load_model(const std::string &filename)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) error->all(FLERR, "Cannot open NN-MTP v2 model file");

  // Magic
  char magic[7] = {0};
  fread(magic, 1, 6, fp);
  if (std::string(magic) != "NNMTP2")
    error->all(FLERR, "Invalid model file (expected NNMTP2 magic)");

  // Species
  fread(&n_species, sizeof(int), 1, fp);
  species_list.resize(n_species);
  for (int i = 0; i < n_species; i++)
    fread(&species_list[i], sizeof(int), 1, fp);

  // Model params
  fread(&n_radial_basis, sizeof(int), 1, fp);
  fread(&n_radial_funcs, sizeof(int), 1, fp);
  double r_min_dummy;
  fread(&r_min_dummy, sizeof(double), 1, fp);  // always 0
  fread(&r_max, sizeof(double), 1, fp);
  fread(&descriptor_dim, sizeof(int), 1, fp);
  fread(&embed_dim, sizeof(int), 1, fp);

  // ZBL
  int zbl_flag;
  fread(&zbl_flag, sizeof(int), 1, fp);
  use_zbl = (zbl_flag != 0);
  if (use_zbl) {
    fread(&zbl_r_inner, sizeof(double), 1, fp);
    fread(&zbl_r_outer, sizeof(double), 1, fp);
  }

  // MLP architecture
  int n_mlp_layers;
  fread(&n_mlp_layers, sizeof(int), 1, fp);
  mlp_dims.resize(n_mlp_layers + 1);
  for (int i = 0; i <= n_mlp_layers; i++)
    fread(&mlp_dims[i], sizeof(int), 1, fp);

  // Radial coefficients
  int rc_size;
  fread(&rc_size, sizeof(int), 1, fp);
  radial_coeffs.resize(rc_size);
  for (int i = 0; i < rc_size; i++)
    fread(&radial_coeffs[i], sizeof(double), 1, fp);

  // Species embedding
  int se_size;
  fread(&se_size, sizeof(int), 1, fp);
  species_embed.resize(se_size);
  for (int i = 0; i < se_size; i++)
    fread(&species_embed[i], sizeof(double), 1, fp);

  // Energy shift
  int es_size;
  fread(&es_size, sizeof(int), 1, fp);
  energy_shift.resize(es_size);
  for (int i = 0; i < es_size; i++)
    fread(&energy_shift[i], sizeof(double), 1, fp);

  // MLP weights and biases
  int n_params;
  fread(&n_params, sizeof(int), 1, fp);
  mlp_weights.resize(n_params / 2);
  mlp_biases.resize(n_params / 2);
  for (int l = 0; l < n_params / 2; l++) {
    int wsize;
    fread(&wsize, sizeof(int), 1, fp);
    mlp_weights[l].resize(wsize);
    for (int i = 0; i < wsize; i++)
      fread(&mlp_weights[l][i], sizeof(double), 1, fp);
    int bsize;
    fread(&bsize, sizeof(int), 1, fp);
    mlp_biases[l].resize(bsize);
    for (int i = 0; i < bsize; i++)
      fread(&mlp_biases[l][i], sizeof(double), 1, fp);
  }

  fclose(fp);

  if (comm->me == 0) {
    printf("NN-MTP v2 loaded: %d species, desc=%d, MLP: ",
           n_species, descriptor_dim);
    for (size_t i = 0; i < mlp_dims.size(); i++)
      printf("%d%s", mlp_dims[i], i < mlp_dims.size()-1 ? " " : "\n");
  }
}

/* ---------------------------------------------------------------------- */

void PairNNMTPv2::chebyshev_basis(double r, double *basis, int n)
{
  // Map r from [0, r_max] to [-1, 1] — NO r_min
  double x = 2.0 * r / r_max - 1.0;
  if (x < -1.0) x = -1.0;
  if (x > 1.0) x = 1.0;
  basis[0] = 1.0;
  if (n > 1) basis[1] = x;
  for (int i = 2; i < n; i++)
    basis[i] = 2.0 * x * basis[i - 1] - basis[i - 2];
}

double PairNNMTPv2::cutoff_fn(double r)
{
  if (r >= r_max) return 0.0;
  double x = r / r_max;
  double t = 1.0 - x * x;
  return t * t;
}

double PairNNMTPv2::silu(double x)
{
  double s = 1.0 / (1.0 + exp(-x));
  return x * s;
}

/* ---------------------------------------------------------------------- */

double PairNNMTPv2::zbl_switching(double r)
{
  if (r <= zbl_r_inner) return 1.0;
  if (r >= zbl_r_outer) return 0.0;
  double x = (r - zbl_r_inner) / (zbl_r_outer - zbl_r_inner);
  return (1.0 - x) * (1.0 - x) * (1.0 - x) * (1.0 + 3.0*x + 6.0*x*x);
}

double PairNNMTPv2::zbl_dswitching(double r)
{
  if (r <= zbl_r_inner || r >= zbl_r_outer) return 0.0;
  double x = (r - zbl_r_inner) / (zbl_r_outer - zbl_r_inner);
  double dx_dr = 1.0 / (zbl_r_outer - zbl_r_inner);
  // d/dx [(1-x)^3 * (1 + 3x + 6x^2)]
  // = -3(1-x)^2(1+3x+6x^2) + (1-x)^3(3+12x)
  // = (1-x)^2 [-3(1+3x+6x^2) + (1-x)(3+12x)]
  // = (1-x)^2 [-3-9x-18x^2 + 3+12x-3x-12x^2]
  // = (1-x)^2 [-30x^2]
  // = -30 x^2 (1-x)^2
  return -30.0 * x * x * (1.0 - x) * (1.0 - x) * dx_dr;
}

double PairNNMTPv2::zbl_energy(double r, int Zi, int Zj)
{
  double c[] = {0.1818, 0.5099, 0.2802, 0.02817};
  double d[] = {-3.2, -0.9423, -0.4028, -0.2016};
  double a = 0.4543 * 0.529 / (pow(Zi, 0.23) + pow(Zj, 0.23));
  double x = r / a;
  double phi = 0.0;
  for (int k = 0; k < 4; k++)
    phi += c[k] * exp(d[k] * x);
  return 14.3996 * Zi * Zj / r * phi;
}

double PairNNMTPv2::zbl_denergy(double r, int Zi, int Zj)
{
  double c[] = {0.1818, 0.5099, 0.2802, 0.02817};
  double d[] = {-3.2, -0.9423, -0.4028, -0.2016};
  double a = 0.4543 * 0.529 / (pow(Zi, 0.23) + pow(Zj, 0.23));
  double x = r / a;
  double phi = 0.0, dphi = 0.0;
  for (int k = 0; k < 4; k++) {
    double e = exp(d[k] * x);
    phi += c[k] * e;
    dphi += c[k] * d[k] / a * e;
  }
  double pref = 14.3996 * Zi * Zj;
  return pref * (-phi / (r * r) + dphi / r);
}

/* ---------------------------------------------------------------------- */

double PairNNMTPv2::mlp_forward(double *input, int input_dim)
{
  int n_layers = mlp_weights.size();
  std::vector<double> current(input, input + input_dim);

  for (int l = 0; l < n_layers; l++) {
    int out_dim = mlp_dims[l + 1];
    int in_dim = mlp_dims[l];
    std::vector<double> output(out_dim, 0.0);

    for (int j = 0; j < out_dim; j++) {
      double val = mlp_biases[l][j];
      for (int k = 0; k < in_dim; k++)
        val += mlp_weights[l][j * in_dim + k] * current[k];
      if (l < n_layers - 1)
        val = silu(val);
      output[j] = val;
    }
    current = output;
  }
  return current[0];
}

/* ---------------------------------------------------------------------- */

int PairNNMTPv2::pack_reverse_comm(int n, int first, double *buf)
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

void PairNNMTPv2::unpack_reverse_comm(int n, int *list, double *buf)
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

/* ======================================================================
   MAIN COMPUTE
   ====================================================================== */

void PairNNMTPv2::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int n_mu = n_radial_funcs;  // 7
  int n_pairs = n_species * n_species;
  double total_eng = 0.0;

  // Per-atom storage for moment tensors
  // M0: [nall, 7], M1: [nall, 3, 3] (mu=0,1,2 only), M2: [nall, 1, 3, 3] (mu=0 only)
  std::vector<double> M0(nall * n_mu, 0.0);
  std::vector<double> M1(nall * 3 * 3, 0.0);  // 3 mu values × 3 xyz
  std::vector<double> M2(nall * 9, 0.0);  // mu=0 only, 3×3 matrix

  // Neighbor data storage
  struct NeighData {
    int j, itype, jtype;
    double dx, dy, dz, r;
    double f_mu[7];       // radial functions
    double df_mu_dr[7];   // d(f_mu)/dr (includes cutoff derivative)
  };
  std::vector<std::vector<NeighData>> all_neigh(inum);

  double basis[16];
  double dbasis_dr[16];

  // ============================================================
  // Pass 1: Compute moment tensors and store neighbor data
  // ============================================================
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int itype = type_map[type[i]];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    all_neigh[ii].reserve(jnum);

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      int jtype = type_map[type[j]];

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];
      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq >= r_max * r_max || rsq < 1e-16) continue;

      double r = sqrt(rsq);
      double ri = 1.0 / r;
      double ux = dx * ri, uy = dy * ri, uz = dz * ri;

      // Chebyshev basis
      chebyshev_basis(r, basis, n_radial_basis);
      double fc = cutoff_fn(r);

      // Cutoff derivative
      double xr = r / r_max;
      double dfc_dr = -4.0 * xr * (1.0 - xr*xr) / r_max;

      // Chebyshev derivative: no r_min, so dx_dr is always valid for r in [0, r_max]
      double dx_dr = (r >= 0.0 && r <= r_max) ? 2.0 / r_max : 0.0;
      double xx = 2.0 * r / r_max - 1.0;
      if (xx < -1.0) xx = -1.0; if (xx > 1.0) xx = 1.0;

      dbasis_dr[0] = 0.0;
      if (n_radial_basis > 1) dbasis_dr[1] = dx_dr;
      for (int b = 2; b < n_radial_basis; b++)
        dbasis_dr[b] = 2.0*(basis[b-1]*dx_dr + xx*dbasis_dr[b-1]) - dbasis_dr[b-2];

      NeighData nd;
      nd.j = j; nd.itype = itype; nd.jtype = jtype;
      nd.dx = dx; nd.dy = dy; nd.dz = dz; nd.r = r;

      int pair_idx = itype * n_species + jtype;

      for (int mu = 0; mu < n_mu; mu++) {
        double fm = 0.0, dfm = 0.0;
        for (int b = 0; b < n_radial_basis; b++) {
          int cidx = mu * n_pairs * n_radial_basis + pair_idx * n_radial_basis + b;
          double c = radial_coeffs[cidx];
          fm += c * basis[b] * fc;
          dfm += c * (dbasis_dr[b] * fc + basis[b] * dfc_dr);
        }
        nd.f_mu[mu] = fm;
        nd.df_mu_dr[mu] = dfm;

        // Accumulate moment tensors
        // M0[i, mu]
        M0[i * n_mu + mu] += fm;

        // M1[i, mu, xyz] for mu=0,1,2
        if (mu < 3) {
          M1[i * 9 + mu * 3 + 0] += fm * ux;
          M1[i * 9 + mu * 3 + 1] += fm * uy;
          M1[i * 9 + mu * 3 + 2] += fm * uz;
        }

        // M2[i, 3x3] for mu=0 only
        if (mu == 0) {
          M2[i * 9 + 0] += fm * ux * ux;
          M2[i * 9 + 1] += fm * ux * uy;
          M2[i * 9 + 2] += fm * ux * uz;
          M2[i * 9 + 3] += fm * uy * ux;
          M2[i * 9 + 4] += fm * uy * uy;
          M2[i * 9 + 5] += fm * uy * uz;
          M2[i * 9 + 6] += fm * uz * ux;
          M2[i * 9 + 7] += fm * uz * uy;
          M2[i * 9 + 8] += fm * uz * uz;
        }
      }

      all_neigh[ii].push_back(nd);
    }
  }

  // ============================================================
  // Pass 2: Compute 27 basis, MLP forward+backward, forces
  // ============================================================
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int itype = type_map[type[i]];

    // Extract moments for atom i
    double m0[7];
    for (int mu = 0; mu < 7; mu++) m0[mu] = M0[i * n_mu + mu];

    double m1[3][3];  // m1[mu][xyz]
    for (int mu = 0; mu < 3; mu++)
      for (int d = 0; d < 3; d++)
        m1[mu][d] = M1[i * 9 + mu * 3 + d];

    double m2[9];  // m2[3x3] flattened, mu=0 only
    for (int k = 0; k < 9; k++) m2[k] = M2[i * 9 + k];

    // Vector dot products
    auto vdot = [&](int a, int b) -> double {
      return m1[a][0]*m1[b][0] + m1[a][1]*m1[b][1] + m1[a][2]*m1[b][2];
    };

    // Matrix Frobenius norm squared
    double m2_frob = 0.0;
    for (int k = 0; k < 9; k++) m2_frob += m2[k] * m2[k];

    // ---- 27 Basis functions ----
    double B[27];
    B[0]  = m0[0];
    B[1]  = m0[0] * m0[0];
    B[2]  = m0[0] * m0[0] * m0[0];
    B[3]  = m0[0] * m0[0] * m0[0] * m0[0];
    B[4]  = m0[1];
    B[5]  = m0[0] * m0[1];
    B[6]  = m0[0] * m0[0] * m0[1];
    B[7]  = m0[1] * m0[1];
    B[8]  = m0[0] * m0[1] * m0[1];
    B[9]  = m0[2];
    B[10] = m0[0] * m0[2];
    B[11] = m0[0] * m0[0] * m0[2];
    B[12] = m0[1] * m0[2];
    B[13] = m0[2] * m0[2];
    B[14] = m0[3];
    B[15] = m0[0] * m0[3];
    B[16] = m0[1] * m0[3];
    B[17] = m0[4];
    B[18] = m0[0] * m0[4];
    B[19] = m0[5];
    B[20] = m0[6];
    B[21] = vdot(0, 0);
    B[22] = m0[0] * vdot(0, 0);
    B[23] = vdot(0, 1);
    B[24] = vdot(0, 2);
    B[25] = vdot(1, 1);
    B[26] = m2_frob;

    // ---- MLP input: 27 descriptors + 16 embedding = 43 ----
    double mlp_input[256];
    for (int k = 0; k < 27; k++) mlp_input[k] = B[k];
    for (int k = 0; k < embed_dim; k++)
      mlp_input[27 + k] = species_embed[itype * embed_dim + k];

    // ---- MLP forward + backward ----
    int n_layers = mlp_weights.size();
    int total_input_dim = mlp_dims[0];

    // Forward pass (save activations)
    std::vector<std::vector<double>> layer_inputs(n_layers + 1);
    layer_inputs[0].assign(mlp_input, mlp_input + total_input_dim);

    for (int l = 0; l < n_layers; l++) {
      int in_dim = mlp_dims[l];
      int out_dim = mlp_dims[l + 1];
      layer_inputs[l + 1].resize(out_dim);
      for (int j = 0; j < out_dim; j++) {
        double val = mlp_biases[l][j];
        for (int k = 0; k < in_dim; k++)
          val += mlp_weights[l][j * in_dim + k] * layer_inputs[l][k];
        if (l < n_layers - 1)
          val = silu(val);
        layer_inputs[l + 1][j] = val;
      }
    }

    double atomic_energy = layer_inputs[n_layers][0] + energy_shift[itype];
    total_eng += atomic_energy;

    // Backward pass: dE/d(input)
    std::vector<double> grad(1, 1.0);  // dE/dE = 1

    for (int l = n_layers - 1; l >= 0; l--) {
      int in_dim = mlp_dims[l];
      int out_dim = mlp_dims[l + 1];
      std::vector<double> new_grad(in_dim, 0.0);

      for (int k = 0; k < in_dim; k++) {
        for (int j = 0; j < out_dim; j++) {
          double g = grad[j] * mlp_weights[l][j * in_dim + k];
          if (l < n_layers - 1) {
            // SiLU derivative
            double pre_act_val = 0.0;
            // Recompute pre-activation
            double pre = mlp_biases[l][j];
            for (int kk = 0; kk < in_dim; kk++)
              pre += mlp_weights[l][j * in_dim + kk] * layer_inputs[l][kk];
            double sig = 1.0 / (1.0 + exp(-pre));
            double dsilu = sig + pre * sig * (1.0 - sig);
            g *= dsilu;
          }
          new_grad[k] += g;
        }
      }
      grad = new_grad;
    }

    // grad[0..26] = dE/dB[0..26] (descriptor gradients)
    double dE_dB[27];
    for (int k = 0; k < 27; k++) dE_dB[k] = grad[k];

    // ---- Compute dE/dM0[mu], dE/dM1[mu][d], dE/dM2[k] from chain rule ----

    // dE/dM0[mu] via B0-B20, B22
    double dE_dm0[7] = {0};

    // B0: m0[0] → dE_dm0[0] += dE_dB[0]
    dE_dm0[0] += dE_dB[0];
    // B1: m0[0]^2 → 2*m0[0]
    dE_dm0[0] += dE_dB[1] * 2 * m0[0];
    // B2: m0[0]^3 → 3*m0[0]^2
    dE_dm0[0] += dE_dB[2] * 3 * m0[0] * m0[0];
    // B3: m0[0]^4 → 4*m0[0]^3
    dE_dm0[0] += dE_dB[3] * 4 * m0[0] * m0[0] * m0[0];
    // B4: m0[1]
    dE_dm0[1] += dE_dB[4];
    // B5: m0[0]*m0[1]
    dE_dm0[0] += dE_dB[5] * m0[1];
    dE_dm0[1] += dE_dB[5] * m0[0];
    // B6: m0[0]^2*m0[1]
    dE_dm0[0] += dE_dB[6] * 2 * m0[0] * m0[1];
    dE_dm0[1] += dE_dB[6] * m0[0] * m0[0];
    // B7: m0[1]^2
    dE_dm0[1] += dE_dB[7] * 2 * m0[1];
    // B8: m0[0]*m0[1]^2
    dE_dm0[0] += dE_dB[8] * m0[1] * m0[1];
    dE_dm0[1] += dE_dB[8] * 2 * m0[0] * m0[1];
    // B9: m0[2]
    dE_dm0[2] += dE_dB[9];
    // B10: m0[0]*m0[2]
    dE_dm0[0] += dE_dB[10] * m0[2];
    dE_dm0[2] += dE_dB[10] * m0[0];
    // B11: m0[0]^2*m0[2]
    dE_dm0[0] += dE_dB[11] * 2 * m0[0] * m0[2];
    dE_dm0[2] += dE_dB[11] * m0[0] * m0[0];
    // B12: m0[1]*m0[2]
    dE_dm0[1] += dE_dB[12] * m0[2];
    dE_dm0[2] += dE_dB[12] * m0[1];
    // B13: m0[2]^2
    dE_dm0[2] += dE_dB[13] * 2 * m0[2];
    // B14: m0[3]
    dE_dm0[3] += dE_dB[14];
    // B15: m0[0]*m0[3]
    dE_dm0[0] += dE_dB[15] * m0[3];
    dE_dm0[3] += dE_dB[15] * m0[0];
    // B16: m0[1]*m0[3]
    dE_dm0[1] += dE_dB[16] * m0[3];
    dE_dm0[3] += dE_dB[16] * m0[1];
    // B17: m0[4]
    dE_dm0[4] += dE_dB[17];
    // B18: m0[0]*m0[4]
    dE_dm0[0] += dE_dB[18] * m0[4];
    dE_dm0[4] += dE_dB[18] * m0[0];
    // B19: m0[5]
    dE_dm0[5] += dE_dB[19];
    // B20: m0[6]
    dE_dm0[6] += dE_dB[20];
    // B22: m0[0] * vdot(0,0)
    dE_dm0[0] += dE_dB[22] * vdot(0, 0);

    // dE/dM1[mu][d] via B21-B25, B22
    double dE_dm1[3][3] = {{0}};  // [mu][xyz]

    // B21: vdot(0,0) = sum_d m1[0][d]^2
    for (int d = 0; d < 3; d++)
      dE_dm1[0][d] += dE_dB[21] * 2 * m1[0][d];
    // B22: m0[0] * vdot(0,0)
    for (int d = 0; d < 3; d++)
      dE_dm1[0][d] += dE_dB[22] * m0[0] * 2 * m1[0][d];
    // B23: vdot(0,1) = sum_d m1[0][d]*m1[1][d]
    for (int d = 0; d < 3; d++) {
      dE_dm1[0][d] += dE_dB[23] * m1[1][d];
      dE_dm1[1][d] += dE_dB[23] * m1[0][d];
    }
    // B24: vdot(0,2)
    for (int d = 0; d < 3; d++) {
      dE_dm1[0][d] += dE_dB[24] * m1[2][d];
      dE_dm1[2][d] += dE_dB[24] * m1[0][d];
    }
    // B25: vdot(1,1)
    for (int d = 0; d < 3; d++)
      dE_dm1[1][d] += dE_dB[25] * 2 * m1[1][d];

    // dE/dM2[k] via B26: m2_frob = sum_k m2[k]^2
    double dE_dm2[9] = {0};
    for (int k = 0; k < 9; k++)
      dE_dm2[k] = dE_dB[26] * 2 * m2[k];

    // ============================================================
    // Pass 3: Compute forces from dE/dM → dE/dr_ij
    // ============================================================
    for (auto &nd : all_neigh[ii]) {
      int j = nd.j;
      double r = nd.r;
      double ri = 1.0 / r;
      double dx = nd.dx, dy = nd.dy, dz = nd.dz;
      double ux = dx * ri, uy = dy * ri, uz = dz * ri;

      double fx_j = 0.0, fy_j = 0.0, fz_j = 0.0;

      for (int mu = 0; mu < n_mu; mu++) {
        double fm = nd.f_mu[mu];
        double dfm_dr = nd.df_mu_dr[mu];

        // --- dE/dr from M0 (scalar moments) ---
        // dM0[mu]/dr_ij = df_mu/dr * r_hat
        double dE_r_m0 = dE_dm0[mu] * dfm_dr;
        fx_j += dE_r_m0 * ux;
        fy_j += dE_r_m0 * uy;
        fz_j += dE_r_m0 * uz;

        // --- dE/dr from M1 (vector moments, mu=0,1,2) ---
        if (mu < 3) {
          // M1[mu][d] = sum_j fm * u_d
          // dM1/dr_j has two terms:
          //   (1) dfm/dr * u_d * r_hat  (radial part)
          //   (2) fm * d(u_d)/dr_j       (angular part)
          // d(u_d)/dr_j = (delta_d - u_d * r_hat) / r

          for (int d = 0; d < 3; d++) {
            double u_d = (d == 0) ? ux : (d == 1) ? uy : uz;
            // Term 1: dfm/dr * u_d → radial
            double t1 = dE_dm1[mu][d] * dfm_dr * u_d;
            fx_j += t1 * ux;
            fy_j += t1 * uy;
            fz_j += t1 * uz;

            // Term 2: fm * (delta_{d,alpha} - u_d * u_alpha) / r
            double fd_x = dE_dm1[mu][d] * fm * ri * ((d == 0 ? 1.0 : 0.0) - u_d * ux);
            double fd_y = dE_dm1[mu][d] * fm * ri * ((d == 1 ? 1.0 : 0.0) - u_d * uy);
            double fd_z = dE_dm1[mu][d] * fm * ri * ((d == 2 ? 1.0 : 0.0) - u_d * uz);
            fx_j += fd_x;
            fy_j += fd_y;
            fz_j += fd_z;
          }
        }

        // --- dE/dr from M2 (matrix moment, mu=0 only) ---
        if (mu == 0) {
          // M2[a*3+b] = sum_j f0 * u_a * u_b
          for (int a = 0; a < 3; a++) {
            double u_a = (a == 0) ? ux : (a == 1) ? uy : uz;
            for (int b = 0; b < 3; b++) {
              double u_b = (b == 0) ? ux : (b == 1) ? uy : uz;
              int idx = a * 3 + b;

              // Term 1: df0/dr * u_a * u_b
              double t1 = dE_dm2[idx] * dfm_dr * u_a * u_b;
              fx_j += t1 * ux;
              fy_j += t1 * uy;
              fz_j += t1 * uz;

              // Term 2: f0 * d(u_a*u_b)/dr_j
              // d(u_a*u_b)/dr_alpha = (delta_a_alpha*u_b + u_a*delta_b_alpha
              //                        - 2*u_a*u_b*u_alpha) / r
              double coeff = dE_dm2[idx] * fm * ri;
              fx_j += coeff * ((a==0?u_b:0) + (b==0?u_a:0) - 2*u_a*u_b*ux);
              fy_j += coeff * ((a==1?u_b:0) + (b==1?u_a:0) - 2*u_a*u_b*uy);
              fz_j += coeff * ((a==2?u_b:0) + (b==2?u_a:0) - 2*u_a*u_b*uz);
            }
          }
        }
      }

      // --- ZBL force ---
      if (use_zbl && r < zbl_r_outer) {
        int Zi = species_list[nd.itype];
        int Zj = species_list[nd.jtype];
        double e_zbl = zbl_energy(r, Zi, Zj);
        double de_zbl_dr = zbl_denergy(r, Zi, Zj);
        double sw = zbl_switching(r);
        double dsw = zbl_dswitching(r);
        // d(e_zbl * sw)/dr = de_zbl*sw + e_zbl*dsw
        double zbl_force_r = (de_zbl_dr * sw + e_zbl * dsw) * 0.5;
        fx_j += zbl_force_r * ux;
        fy_j += zbl_force_r * uy;
        fz_j += zbl_force_r * uz;
      }

      // Apply forces: f[i] += F, f[j] -= F
      f[i][0] += fx_j;
      f[i][1] += fy_j;
      f[i][2] += fz_j;

      // Ghost → local mapping
      int real_j = atom->map(atom->tag[nd.j]);
      if (real_j >= 0 && real_j < nlocal) {
        f[real_j][0] -= fx_j;
        f[real_j][1] -= fy_j;
        f[real_j][2] -= fz_j;
      }
    }

    // ZBL energy contribution
    if (use_zbl) {
      for (auto &nd : all_neigh[ii]) {
        if (nd.r < zbl_r_outer) {
          int Zi = species_list[nd.itype];
          int Zj = species_list[nd.jtype];
          double e_zbl = zbl_energy(nd.r, Zi, Zj);
          double sw = zbl_switching(nd.r);
          total_eng += 0.5 * e_zbl * sw;
        }
      }
    }
  }

  // Reverse comm for ghost forces
  comm->reverse_comm(this);

  if (eflag_global) eng_vdwl += total_eng;
  if (eflag_atom) {
    // Per-atom energy already accumulated in total_eng
    // For simplicity, distribute evenly (TODO: improve)
  }
}
