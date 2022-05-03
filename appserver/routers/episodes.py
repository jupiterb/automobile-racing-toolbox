from fastapi import APIRouter, Response, status

from episodes.episode_recording_manager import EpisodeRecordingManager
from schemas import Episode
from routers.common import Repositories


episodes_router = APIRouter(
    prefix="/games/{game_id}/episodes",
    tags=["episodes"]
)

games = Repositories.games
episodes = Repositories.episodes
episodes_recording_manager = EpisodeRecordingManager()


@episodes_router.get("/")
async def get_episodes(game_id: str) -> list[Episode]:
    return episodes.get_all(lambda game_episode_id: game_episode_id[0] == game_id)


@episodes_router.get("/{episode_id}")
async def get_episode(game_id: str, episode_id: str) -> Episode:
    return episodes.get_item((game_id, episode_id))


@episodes_router.post("/{episode_id}", status_code=status.HTTP_201_CREATED)
async def add_episode(game_id: str, episode_id: str, description: str, response: Response) -> Episode:
    new_episode = Episode(id=episode_id, description=description)
    created, returned_episode = episodes.add_item((game_id, episode_id), new_episode)
    if not created:
        response.status_code = status.HTTP_200_OK
    return returned_episode
    

@episodes_router.delete("/{episode_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_episode(game_id: str, episode_id: str):
    episodes.delete_item((game_id, episode_id))


@episodes_router.get("/{episode_id}/record")
async def run_episode_recording(
        game_id: str, 
        episode_id: str, 
        fps: int = EpisodeRecordingManager.default_fps()
    ) -> Episode:
    game = games.get_item(game_id)
    recording = episodes_recording_manager.__record(
        game.system_configuration, 
        game.global_configuration,
        fps
    )
    return episodes.update_item((game_id, episode_id), recording=recording)
    