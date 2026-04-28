from action_correct import action_correct
from checkpoint_correct import checkpoint_correct
from command_correct import command_correct
from logs_correct import logs_correct
from loss_correct import loss_correct
from model_correct import model_correct
from observation_correct import observation_correct
from project_analysis import ProjectAnalysis
from reset_correct import reset_correct
from reward_correct import reward_correct
from scene_correct import scene_correct
from trainer_correct import trainer_correct
from viewer_correct import viewer_correct


project_analysis = ProjectAnalysis(
    scene_correct=scene_correct,
    model_correct=model_correct,
    action_correct=action_correct,
    observation_correct=observation_correct,
    reset_correct=reset_correct,
    reward_correct=reward_correct,
    loss_correct=loss_correct,
    trainer_correct=trainer_correct,
    logs_correct=logs_correct,
    checkpoint_correct=checkpoint_correct,
    viewer_correct=viewer_correct,
    command_correct=command_correct,
)
