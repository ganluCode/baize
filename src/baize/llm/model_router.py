"""Agent-level model routing for Baize.

Resolves which LLM instance to use for a given agent and usage type
(chat / reasoning / embedding), with fallback logic when an agent or
usage type is not explicitly configured.
"""

import logging

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from baize.llm.provider import ProviderFactory
from baize.llm.schemas import AgentConfig

logger = logging.getLogger(__name__)

_DEFAULT_AGENT = "default"


class ModelRouter:
    """Routes model requests to the correct provider based on agent configuration.

    Args:
        provider_factory: Initialised factory used to retrieve model instances.
        agents: List of agent configurations, each binding a name to model refs.
    """

    def __init__(self, provider_factory: ProviderFactory, agents: list[AgentConfig]) -> None:
        self._factory = provider_factory
        self._agents: dict[str, AgentConfig] = {a.name: a for a in agents}

    def _get_agent(self, agent_name: str) -> AgentConfig:
        """Return the named agent, falling back to 'default' if not found."""
        if agent_name in self._agents:
            return self._agents[agent_name]
        if agent_name != _DEFAULT_AGENT and _DEFAULT_AGENT in self._agents:
            logger.debug("Agent '%s' not found; falling back to '%s'.", agent_name, _DEFAULT_AGENT)
            return self._agents[_DEFAULT_AGENT]
        raise KeyError(f"Agent '{agent_name}' not found and no '{_DEFAULT_AGENT}' agent is configured.")

    @staticmethod
    def _parse_ref(ref: str) -> tuple[str, str]:
        """Split 'provider_name/model_id' into a (provider_name, model_id) tuple."""
        provider_name, model_id = ref.split("/", 1)
        return provider_name, model_id

    def get_chat_model(self, agent_name: str) -> ChatOpenAI | ChatAnthropic:
        """Return the chat model for the given agent.

        Args:
            agent_name: Name of the agent. Falls back to 'default' if unknown.

        Returns:
            A ``ChatOpenAI`` or ``ChatAnthropic`` instance.
        """
        agent = self._get_agent(agent_name)
        provider_name, model_id = self._parse_ref(agent.models.chat)  # type: ignore[arg-type]
        return self._factory.get_chat_model(provider_name, model_id)

    def get_reasoning_model(self, agent_name: str) -> ChatOpenAI | ChatAnthropic:
        """Return the reasoning model for the given agent.

        Falls back to the agent's chat model when ``reasoning`` is not configured.

        Args:
            agent_name: Name of the agent. Falls back to 'default' if unknown.

        Returns:
            A ``ChatOpenAI`` or ``ChatAnthropic`` instance.
        """
        agent = self._get_agent(agent_name)
        ref = agent.models.reasoning or agent.models.chat
        provider_name, model_id = self._parse_ref(ref)  # type: ignore[arg-type]
        return self._factory.get_chat_model(provider_name, model_id)

    def get_embedding_model(self, agent_name: str) -> OpenAIEmbeddings:
        """Return the embedding model for the given agent.

        Falls back to the agent's chat model ref (calling get_embedding_model) when
        ``embedding`` is not configured.

        Args:
            agent_name: Name of the agent. Falls back to 'default' if unknown.

        Returns:
            An ``OpenAIEmbeddings`` instance.
        """
        agent = self._get_agent(agent_name)
        ref = agent.models.embedding or agent.models.chat
        provider_name, model_id = self._parse_ref(ref)  # type: ignore[arg-type]
        return self._factory.get_embedding_model(provider_name, model_id)
