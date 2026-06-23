/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS pair_style for NN-MTP (Neural Network Moment Tensor Potential)
   Pure C++ implementation — no LibTorch, no Python
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nnmtp,PairNNMTP);
// clang-format on
#else

#ifndef LMP_PAIR_NNMTP_H
#define LMP_PAIR_NNMTP_H

#include "pair.h"
#include <vector>
#include <string>

namespace LAMMPS_NS {

class PairNNMTP : public Pair {
 public:
  PairNNMTP(class LAMMPS *);
  ~PairNNMTP() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  // For reverse_comm: ghost forces → local atoms
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  int n_species;
  int n_radial_basis;
  int n_radial_funcs;
  int descriptor_dim;
  double r_min, r_max;
  std::vector<int> hidden_dims;

  std::vector<double> radial_coeffs;
  std::vector<double> species_embed;
  int embed_dim;
  std::vector<std::vector<double>> mlp_weights;
  std::vector<std::vector<double>> mlp_biases;
  std::vector<int> mlp_dims;
  std::vector<double> energy_shift;
  std::vector<int> type_map;
  bool use_zbl;
  double zbl_r_inner, zbl_r_outer;
  int species_Z[4];  // atomic numbers for Li, P, S, Cl

  void load_model(const std::string &filename);
  void chebyshev_basis(double r, double *basis, int n);
  double cutoff_fn(double r);
  double silu(double x);
  void zbl_pair(double r, int Z_i, int Z_j, double &e_zbl, double &de_zbl_dr);
  void compute_descriptors(int i, int itype, int *jlist, int jnum, double *desc);
  double mlp_forward(double *input, int input_dim);
  void allocate();
};

}  // namespace LAMMPS_NS

#endif
#endif
