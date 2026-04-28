from command_leaps import command_leaps
from enhance_guidance import EnhanceGuidance
from environment_leaps import environment_leaps
from model_leaps import model_leaps
from observation_leaps import observation_leaps
from other_leaps import other_leaps
from reward_leaps import reward_leaps
from training_leaps import training_leaps


enhance_guidance = EnhanceGuidance(
    model_leaps=model_leaps,
    training_leaps=training_leaps,
    command_leaps=command_leaps,
    environment_leaps=environment_leaps,
    observation_leaps=observation_leaps,
    reward_leaps=reward_leaps,
    other_leaps=other_leaps,
)
