/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS pair_style for NN-MTP v2
   Pure C++ implementation — no LibTorch, no Python

   v2 changes: 27 basis (LightPFP), ZBL, no r_min, atomic number based
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nnmtp2,PairNNMTPv2);
// clang-format on
#else

#ifndef LMP_PAIR_NNMTP_V2_H
#define LMP_PAIR_NNMTP_V2_H

#include "pair.h"
#include <vector>
#include <string>

namespace LAMMPS_NS {

class PairNNMTPv2 : public Pair {
 public:
  PairNNMTPv2(class LAMMPS *);
  ~PairNNMTPv2() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  // Model parameters
  int n_species;
  std::vector<int> species_list;  // atomic numbers
  int n_radial_basis;
  int n_radial_funcs;  // always 7 for v2
  int descriptor_dim;  // always 27 for v2
  double r_max;
  int embed_dim;

  // ZBL
  bool use_zbl;
  double zbl_r_inner, zbl_r_outer;

  // MLP
  std::vector<int> hidden_dims;
  std::vector<std::vector<double>> mlp_weights;
  std::vector<std::vector<double>> mlp_biases;
  std::vector<int> mlp_dims;

  // Learned parameters
  std::vector<double> radial_coeffs;
  std::vector<double> species_embed;
  std::vector<double> energy_shift;
  std::vector<int> type_map;  // LAMMPS type → species index

  void load_model(const std::string &filename);
  void chebyshev_basis(double r, double *basis, int n);
  double cutoff_fn(double r);
  double silu(double x);
  double zbl_energy(double r, int Zi, int Zj);
  double zbl_denergy(double r, int Zi, int Zj);
  double zbl_switching(double r);
  double zbl_dswitching(double r);
  double mlp_forward(double *input, int input_dim);
  void allocate();
};

}  // namespace LAMMPS_NS

#endif
#endif
