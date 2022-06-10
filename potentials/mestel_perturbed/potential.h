#ifndef SWIFT_POTENTIAL_MESTEL_H
#define SWIFT_POTENTIAL_MESTEL_H

/* Config parameters. */
#include "../config.h"

/* Some standard headers. */
#include <float.h>
#include <math.h>

/* Local includes. */
#include "error.h"
#include "gravity.h"
#include "parser.h"
#include "part.h"
#include "physical_constants.h"
#include "space.h"
#include "units.h"

/**
 * @brief External Potential Properties - Mestel Disk
 * [TODO]
 */
struct external_potential {
  /*! Position of the Mestel potential */
  double x[3];

  /*! Circular Velocity */
  double v0;

  /*! Scale Radius */
  double r0;

  /*! Time-step condition pre-factor */
  float timestep_mult;

  /*! Minimum time step based on the circular orbital time at the
  *  scale raidus times the timestep_mult */
  double mintime;

  /*! Perturbation initial radius & angle */
  double pert_r0;
  double pert_alpha0;
  double pert_v0;

  /*! Perturber Mass */
  double pert_mass;

  /*! Perturber softening */
  double pert_softening;

  /*! Perturbation angular speed & coefficients (convenience) */
  double omega;
  double A;
  double B;
  double C;
  double D;
};

/**
 * @brief Computes the time-step due to the acceleration from a Mestel potential
 *
 * Given by a specified fraction of circular orbital time at radius of particle
 *
 * @param time The current time.
 * @param potential The properties of the externa potential.
 * @param phys_const The physical constants in internal units.
 * @param g Pointer to the g-particle data.
 */
__attribute__((always_inline)) INLINE static float external_gravity_timestep(
    double time, const struct external_potential* restrict potential,
    const struct phys_const* restrict phys_const,
    const struct gpart* restrict g) {

  const float dx = g->x[0] - potential->x[0];
  const float dy = g->x[1] - potential->x[1];
  const float dz = g->x[2] - potential->x[2];
  const float r = sqrtf(dx * dx + dy * dy + dz * dz);
  const float time_step = potential->timestep_mult * r/potential->v0;

  // Perturber (copied from point_mass_softened)
  const float pert_dx = g->x[0] - potential->A*sin(potential->omega*time) - potential->B*cos(potential->omega*time) - potential->x[0];
  const float pert_dy = g->x[1] - potential->C*sin(potential->omega*time) - potential->D*cos(potential->omega*time) - potential->x[1];
  const float pert_dz = g->x[2]- potential->x[2];

  const float softening2 = potential->pert_softening * potential->pert_softening;
  const float r2 = pert_dx * pert_dx + pert_dy * pert_dy + pert_dz * pert_dz;
  const float rinv = 1.f / sqrtf(r2);
  const float rinv2_softened = 1.f / (r2 + softening2);

  /* G * M / (r^2 + eta^2)^{3/2} */
  const float GMr32 = phys_const->const_newton_G * potential->pert_mass *
                      sqrtf(rinv2_softened * rinv2_softened * rinv2_softened);

  /* This is actually dr dot v */
  const float drdv =
      (pert_dx) * (g->v_full[0]) + (pert_dy) * (g->v_full[1]) + (pert_dz) * (g->v_full[2]);

  /* We want da/dt */
  /* da/dt = -GM/(r^2 + \eta^2)^{3/2}  *
   *         (\vec{v} - 3 \vec{r} \cdot \vec{v} \hat{r} /
   *         (r^2 + \eta^2)) */
  const float dota_x =
      GMr32 * (3.f * drdv * pert_dx * rinv2_softened * rinv - g->v_full[0]);
  const float dota_y =
      GMr32 * (3.f * drdv * pert_dy * rinv2_softened * rinv - g->v_full[0]);
  const float dota_z =
      GMr32 * (3.f * drdv * pert_dz * rinv2_softened * rinv - g->v_full[0]);

  const float dota_2 = dota_x * dota_x + dota_y * dota_y + dota_z * dota_z;
  const float a_2 = g->a_grav[0] * g->a_grav[0] + g->a_grav[1] * g->a_grav[1] +
                    g->a_grav[2] * g->a_grav[2];

  // *10 factor since time timestep_mult is in general 10 times larger for the point mass condition (see swift ext pots)
  const float pert_time_step = 10.0 * potential->timestep_mult * sqrtf(a_2 / dota_2);

  const float maxts = max(time_step,pert_time_step);
  return max(maxts, potential->mintime);
  //return max(time_step, potential->mintime);
}

/**
 * @brief Computes the gravitational acceleration of a particle due to a
 * Mestel potential
 *
 * Note that the accelerations are multiplied by Newton's G constant later
 * on.
 *
 * We pass in the time for simulations where the potential evolves with time.
 *
 * @param time The current time.
 * @param potential The proerties of the external potential.
 * @param phys_const The physical constants in internal units.
 * @param g Pointer to the g-particle data.
 */
__attribute__((always_inline)) INLINE static void external_gravity_acceleration(
    double time, const struct external_potential* restrict potential,
    const struct phys_const* restrict phys_const, struct gpart* restrict g) {

  const float dx = g->x[0] - potential->x[0];
  const float dy = g->x[1] - potential->x[1];
  const float dz = g->x[2] - potential->x[2];

  const float v02 = potential->v0*potential->v0;
  const float r = sqrtf(dx * dx + dy * dy + dz * dz);
  const float rinv2G = 1.f / ((dx * dx + dy * dy + dz * dz)*phys_const->const_newton_G);
  const float pot = v02 * logf(r/potential->r0) / phys_const->const_newton_G;

  g->a_grav[0] += -v02 * dx * rinv2G;
  g->a_grav[1] += -v02 * dy * rinv2G;
  g->a_grav[2] += -v02 * dz * rinv2G;

  // Perturbation (copied)
  const float pert_dx = g->x[0] - potential->A*sin(potential->omega*time) - potential->B*cos(potential->omega*time) - potential->x[0];
  const float pert_dy = g->x[1] - potential->C*sin(potential->omega*time) - potential->D*cos(potential->omega*time) - potential->x[1];
  const float pert_dz = g->x[2]- potential->x[2];
  const float rinv = 1.f / sqrtf(pert_dx * pert_dx + pert_dy * pert_dy + pert_dz * pert_dz +
                                 potential->pert_softening * potential->pert_softening);
  const float rinv3 = rinv * rinv * rinv;

  g->a_grav[0] += -potential->pert_mass * pert_dx * rinv3;
  g->a_grav[1] += -potential->pert_mass * pert_dy * rinv3;
  g->a_grav[2] += -potential->pert_mass * pert_dz * rinv3;

  gravity_add_comoving_potential(g, pot - potential->pert_mass * rinv);
}

/**
 * @brief Computes the gravitational potential energy of a particle in a point
 * Mestel potential.
 *
 * @param time The current time (unused here).
 * @param potential The #external_potential used in the run.
 * @param phys_const Physical constants in internal units.
 * @param g Pointer to the particle data.
 */
__attribute__((always_inline)) INLINE static float
external_gravity_get_potential_energy(
    double time, const struct external_potential* potential,
    const struct phys_const* const phys_const, const struct gpart* g) {

  const float dx = g->x[0] - potential->x[0];
  const float dy = g->x[1] - potential->x[1];
  const float dz = g->x[2] - potential->x[2];
  const float r = sqrtf(dx * dx + dy * dy + dz * dz);

  // Perturber (copied)
  const float pert_dx = g->x[0] - potential->A*sin(potential->omega*time) - potential->B*cos(potential->omega*time) - potential->x[0];
  const float pert_dy = g->x[1] - potential->C*sin(potential->omega*time) - potential->D*cos(potential->omega*time) - potential->x[1];
  const float pert_dz = g->x[2]- potential->x[2];
  const float rinv = 1. / sqrtf(pert_dx * pert_dx + pert_dy * pert_dy + pert_dz * pert_dz +
                                potential->pert_softening * potential->pert_softening);

  return potential->v0*potential->v0 * logf(r/potential->r0) - phys_const->const_newton_G * potential->pert_mass * rinv;;
}

/**
 * @brief Initialises the external potential properties in the internal system
 * of units.
 *
 * @param parameter_file The parsed parameter file
 * @param phys_const Physical constants in internal units
 * @param us The current internal system of units
 * @param s The #space we run in.
 * @param potential The external potential properties to initialize
 */
static INLINE void potential_init_backend(
    struct swift_params* parameter_file, const struct phys_const* phys_const,
    const struct unit_system* us, const struct space* s,
    struct external_potential* potential) {

  /* Read in the position of the centre of potential */
  parser_get_param_double_array(parameter_file, "MestelPotentialPert:position",
                                3, potential->x);

  /* Read the other parameters of the model */
  potential->v0 =
      parser_get_param_double(parameter_file, "MestelPotentialPert:v0");
  potential->timestep_mult = parser_get_opt_param_float(
      parameter_file, "MestelPotentialPert:timestep_mult", FLT_MAX);
  potential->r0 =
      parser_get_param_float(parameter_file, "MestelPotentialPert:r0");
  potential->pert_mass = 
      parser_get_param_float(parameter_file, "MestelPotentialPert:pert_mass");
  potential->pert_softening = 
      parser_get_param_float(parameter_file, "MestelPotentialPert:pert_softening");
  potential->mintime = potential->timestep_mult * potential->r0 / potential->v0;

  /* Read in the relative initial position of the perturber and initialize parameters */
  potential->pert_r0 =
      parser_get_param_double(parameter_file,"MestelPotentialPert:pert_r0");
  potential->pert_v0 =
      parser_get_param_double(parameter_file,"MestelPotentialPert:pert_v0");
  potential->pert_alpha0 =
      parser_get_param_double(parameter_file,"MestelPotentialPert:pert_alpha0"); 

  potential->omega = potential->pert_v0 / potential->pert_r0; /// 3.0857e16 / us->UnitTime_in_cgs; // convert km->kpc
//  potential->A = potential->v0 * cos(potential->pert_alpha0) / potential->omega;
//  potential->C = potential->v0 * sin(potential->pert_alpha0) / potential->omega;
//  potential->B = potential->pert_r0 * cos(potential->pert_alpha0) + potential->x[0];
//  potential->D = potential->pert_r0 * sin(potential->pert_alpha0) + potential->x[1];
  potential->A = potential->pert_r0*sin(potential->pert_alpha0);
  potential->B = potential->pert_r0*cos(potential->pert_alpha0);
  potential->C = potential->pert_r0*cos(potential->pert_alpha0);
  potential->D = potential->pert_r0*sin(potential->pert_alpha0);
}

/**
 * @brief Prints the properties of the external potential to stdout.
 *
 * @param  potential The external potential properties.
 */
static INLINE void potential_print_backend(
    const struct external_potential* potential) {

  message(
      "External potential is 'Mestel Potential + Perturbing mass' with properties (x,y,z) = (%e, %e, "
      "%e), v0 = %e, r0 = %e timestep multiplier = %e, Perturber: r0 = %e, v0= %e, alpha0 = %e"
      "mass = %e, softening = %e",
      potential->x[0], potential->x[1], potential->x[2], potential->v0,
      potential->r0, potential->timestep_mult,
      potential->pert_r0, potential->pert_v0,potential->pert_alpha0,
      potential->pert_mass,potential->pert_softening);
//potential->pert_x[0], potential->pert_x[1], potential->pert_x[2],
}

#endif /* SWIFT_POTENTIAL_MESTEL_H */
