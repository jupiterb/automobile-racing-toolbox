from ui_app.src.forms.game_selection import configure_game
from ui_app.src.forms.env_selection import configure_env
from ui_app.src.forms.training_config_selection import configure_training
from ui_app.src.forms.training_briefing import (
    start_training,
    resume_training,
    train_autoencoder,
)
from ui_app.src.forms.review_config import review_config
from ui_app.src.forms.review_training_tasks import review_tasks
from ui_app.src.forms.wandb_checkpoint_selection import configure_training_resuming
from ui_app.src.forms.user_settings import review_account_settings
from ui_app.src.forms.vae_selection import configure_encoder, configure_vae_training
from ui_app.src.forms.authorization import log_in
