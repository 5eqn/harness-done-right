from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

task_registry = File(
    path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py"
)
g1_env_cfg = File(
    path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py"
)
tracking_env_cfg = File(
    path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py"
)
g1_constants = File(
    path="mjlab/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py"
)
action_manager_py = File(path="mjlab/src/mjlab/managers/action_manager.py")
actions_py = File(path="mjlab/src/mjlab/envs/mdp/actions/actions.py")
manager_env = File(path="mjlab/src/mjlab/envs/manager_based_rl_env.py")

action_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=task_registry,
            content="""
The scoped MJLab task is exactly `Mjlab-Tracking-Flat-Unitree-G1`. It calls `unitree_g1_flat_tracking_env_cfg()` without the no-state-estimation override, so it uses the function's default state-estimation path; the separate `No-State-Estimation` task is a different task id and is outside this proof.""",
            text="""
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The G1 tracking config default is `has_state_estimation=True`. For the scoped task above, no override is passed, so the actor observations keep the state-estimation terms.""",
            text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
The base tracking environment has exactly one configured action term: `joint_pos`. It is a `JointPositionActionCfg` over actuator_names `(".*",)`, has the default joint-position offset enabled, and starts with a placeholder scale that the G1-specific config overrides.""",
            text="""
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The Unitree G1 environment narrows that one action term to `JointPositionActionCfg` and replaces the base scale with `G1_ACTION_SCALE`.""",
            text="""
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE""",
        ),
        Highlight(
            file=g1_constants,
            content="""
`G1_ACTION_SCALE` is not a scalar fallback; it is built per G1 position actuator from actuator effort and stiffness and keyed by each actuator target name.""",
            text="""
G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s""",
        ),
        Highlight(
            file=actions_py,
            content="""
The action dimension for `joint_pos` is exactly the number of targets matched by `actuator_names`. For `(".*",)` with joint transmission, those targets are all actuated G1 joints resolved through `find_joints_by_actuator_names`.""",
            text="""
    # Find targets based on transmission type.
    target_ids, target_names = self._find_targets(cfg)
    self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
    self._target_names = target_names

    self._num_targets = len(target_ids)
    self._action_dim = len(target_ids)""",
        ),
        Highlight(
            file=action_manager_py,
            content="""
The `ActionManager` shape is the sum of term dimensions. Because the scoped config has one term, the flat policy action must have exactly the `joint_pos` dimension.""",
            text="""
  @property
  def total_action_dim(self) -> int:
    return sum(self.action_term_dim)

  @property
  def action_term_dim(self) -> list[int]:
    return [term.action_dim for term in self._terms.values()]""",
        ),
        Highlight(
            file=action_manager_py,
            content="""
`process_action` enforces the shape, records current/previous/two-steps-ago raw action history, and slices the flat action tensor into each term. With one `joint_pos` term, the only slice is the full action vector.""",
            text="""
    if self.total_action_dim != action.shape[1]:
      raise ValueError(
        f"Invalid action shape, expected: {self.total_action_dim},"
        f" received: {action.shape[1]}."
      )
    # Shift history: prev_prev ← prev ← current ← new.
    self._prev_prev_action[:] = self._prev_action
    self._prev_action[:] = self._action
    self._action[:] = action.to(self.device)
    # Split the flat action vector and route each slice to its term.
    idx = 0
    for term in self._terms.values():
      term_actions = action[:, idx : idx + term.action_dim]
      term.process_actions(term_actions)
      idx += term.action_dim""",
        ),
        Highlight(
            file=actions_py,
            content="""
`BaseAction.process_actions` is the concrete affine processing path: raw action is saved, multiplied by the configured scale, shifted by offset, and optionally clipped after processing.""",
            text="""
  def process_actions(self, actions: torch.Tensor):
    \"\"\"Process raw actions by applying scale, offset, and optional clip.\"\"\"
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )""",
        ),
        Highlight(
            file=actions_py,
            content="""
`JointPositionAction` honors `use_default_offset=True` by replacing the base offset with each target joint's default position. Applying the term then subtracts encoder bias before writing the target into the entity.""",
            text="""
class JointPositionAction(BaseAction):
  \"\"\"Control joints via position targets.\"\"\"

  def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)""",
        ),
        Highlight(
            file=manager_env,
            content="""
At runtime, `env.step` processes the policy action once, then applies the most recently processed joint-position targets on every physics substep in the decimation loop.""",
            text="""
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()""",
        ),
    ]
)
