import numpy as np
from typing import Iterable

from dm_control import suite
from dm_control.mujoco.wrapper.mjbindings import mjlib


def get_body_jacobian(physics: suite.manipulator.Physics,
                      body_id: int,
                      body_pos,
                      update_physics: bool = False) -> [np.ndarray, np.ndarray, np.ndarray]:
    # Set the data type for the jacobian
    dtype = physics.data.qpos.dtype

    # Initialize the Jacobian Matrices for Position and Rotation
    jac = np.empty((6, physics.model.nv), dtype=dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]

    if update_physics:
        # Ensure that the position of the site is up to date
        mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

    # Get the Jacobian from Mujoco
    # NOTE: Since Mujoco usually works with 6D poses,
    #       we get a two 3 x Num Joints Jacobians for the position and orientation
    #       For the case of the planar manipulator, we only really need the rows corresponding to the x and z coords
    #       and for the orientation we need only the row corresponding to the pitch. The rest should all be 0 anyways
    #       Additionally, for each row, the only non-zero values should correspond to the joints preceding the site
    #       We leave the handling of this situation to the end user, we return the Jacobians as computed by Mujoco
    mjlib.mj_jac(physics.model.ptr,
                 physics.data.ptr,
                 jac_pos,
                 jac_rot,
                 body_pos,
                 body_id)

    # Stack the position and rotation Jacobians to form the full Jacobian
    jac_full = np.vstack([jac_pos, jac_rot])

    # Return the result
    return jac_full, jac_pos, jac_rot


def get_site_jacobian(physics: suite.manipulator.Physics,
                      site_id: int,
                      update_physics: bool = False) -> [np.ndarray, np.ndarray, np.ndarray]:
    # Set the data type for the jacobian
    dtype = physics.data.qpos.dtype

    # Initialize the Jacobian Matrices for Position and Rotation
    jac = np.empty((6, physics.model.nv), dtype=dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]

    # Make a copy of the physics model (do not modify it in-place)
    # physics = physics.copy(share_model=True)

    if update_physics:
        # Ensure that the position of the site is up to date
        mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

    # Get the Jacobian from Mujoco
    # NOTE: Since Mujoco usually works with 6D poses,
    #       we get a two 3 x Num Joints Jacobians for the position and orientation
    #       For the case of the planar manipulator, we only really need the rows corresponding to the x and z coords
    #       and for the orientation we need only the row corresponding to the pitch. The rest should all be 0 anyways
    #       Additionally, for each row, the only non-zero values should correspond to the joints preceding the site
    #       We leave the handling of this situation to the end user, we return the Jacobians as computed by Mujoco
    mjlib.mj_jacSite(physics.model.ptr,
                     physics.data.ptr,
                     jac_pos,
                     jac_rot,
                     site_id)

    # Stack the position and rotation Jacobians to form the full Jacobian
    jac_full = np.vstack([jac_pos, jac_rot])

    # Return the result
    return jac_full, jac_pos, jac_rot


def get_mass_matrix(physics: suite.manipulator.Physics,
                    update_physics: bool = False) -> np.ndarray:
    # If required, ensure physics/states are up to date
    if update_physics:
        physics.forward()

    # Set the data type for the inertia
    dtype = np.float64

    # Initialize the inertia matrix (in column order format, which Mujoco uses)
    mass_matrix = np.ndarray(shape=(physics.model.nv ** 2,), dtype=dtype, order='C')

    # Form the dense inertia matrix
    mjlib.mj_fullM(physics.model.ptr, mass_matrix, physics.data.qM)

    # Reshape to a square matrix
    mass_matrix = np.reshape(mass_matrix, (physics.model.nv, physics.model.nv))

    return mass_matrix


def get_jnt_ids(physics: suite.manipulator.Physics,
                joints: Iterable[str]):
    # Get ids of joints
    jnt_ids = [physics.model.jnt_qposadr[physics.model.name2id(joint, 'joint')] for joint in joints]

    return jnt_ids


def get_jnt_qpos(physics: suite.manipulator.Physics,
                 joints: Iterable[str]):
    # Get ids of joints
    jnt_ids = get_jnt_ids(physics, joints)

    # Get qpos and return
    qpos = list(map(lambda jnt_id: physics.data.qpos[jnt_id], jnt_ids))

    return qpos


def get_jnt_qacc(physics: suite.manipulator.Physics,
                 joints: Iterable[str]):
    # Get ids of joints
    jnt_ids = get_jnt_ids(physics, joints)

    # Get qacc and return
    qacc = list(map(lambda jnt_id: physics.data.qacc[jnt_id], jnt_ids))

    return qacc
