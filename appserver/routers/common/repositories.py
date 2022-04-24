from repository import GuardRepository, InMemoryRepository
from schemas import Game, Training, Episode


class Repositories:

    games: InMemoryRepository[str, Game] = InMemoryRepository()

    trainings: GuardRepository[str, tuple[str, str], Training] = GuardRepository(
        InMemoryRepository[tuple[str, str], Training](),
        games,
        lambda game_training_id: game_training_id[0]
    )
    
    episodes: GuardRepository[str, tuple[str, str], Episode] = GuardRepository(
        InMemoryRepository[tuple[str, str], Episode](),
        games,
        lambda game_episode_id: game_episode_id[0]
    )
