import browser from "webextension-polyfill"
import { Agent } from "./agent"

export const findAllAgents = async (): Promise<Agent[]> => {
  let { agents } = await browser.storage.local.get("agents")
  if (!agents) {
    agents = []
    await updateAgents(agents)
  }
  return agents.map((a: any) => Agent.fromJsonObject(a))
}

export const findAgentById = async (agentId: string): Promise<Agent | undefined> => {
  let agents = await findAllAgents()
  return agents.find(a => a.manifest.id === agentId)
}

export const addAgent = async (agent: Agent): Promise<void> => {
  let agents = await findAllAgents()
  if (agents.find(a => a.manifest.id === agent.manifest.id)) {
    throw Error("You have already configured a copilot with same ID. You should remove it or verify if your new copilot needs to change its ID in manifest.json")
  }
  agents.push(agent)
  agents.sort((a1, a2) => a1.manifest.id < a2.manifest.id ? -1 : 1)
  updateAgents(agents)
}

const updateAgents = async (agents: Agent[]) => {
  await browser.storage.local.set({ agents: agents })
}

export const removeAgent = async (agentId: string): Promise<void> => {
  let agents = await findAllAgents()
  agents = agents.filter(a => a.manifest.id === agentId)
  updateAgents(agents)
}
