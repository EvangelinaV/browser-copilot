import browser from "webextension-polyfill"
import { Prompt } from "./prompt-repository"
import { AuthService, AuthConfig } from "./auth"
import { fetchJson, fetchStreamJson } from "./http"


export class Agent {
    url: string
    logo: string
    manifest: AgentManifest
    activationRule: AgentRule
    activationAction: ActivationAction

    private constructor(url: string, manifest: AgentManifest) {
        this.url = url
        this.logo = `${url}/logo.png`
        this.manifest = manifest
        this.activationRule = manifest.onHttpRequest?.find(r => r.actions.find(a => a.activate))!
        this.activationAction = this.activationRule?.actions.find(a => a.activate)!.activate!
    }

    public static async fromUrl(url: string): Promise<Agent> {
        url = url.endsWith("/") ? url.slice(0, -1) : url
        let manifestPath = "/manifest.json"
        url = url.endsWith(manifestPath) ? url.slice(0, -manifestPath.length) : url
        return new Agent(url, await fetchJson(`${url}/manifest.json`))
    }

    public static fromJsonObject(obj: any): Agent {
        return new Agent(obj.url, obj.manifest)
    }

    public async createSession(locales: string[], authService?: AuthService): Promise<AgentSession> {
        if (authService) {
            await authService.login()
        }
        return await this.postJson(`${this.url}/sessions`, { locales: locales }, authService)
    }

    private async postJson(url: string, body: any, authService?: AuthService): Promise<any> {
        return await fetchJson(url, await this.buildHttpPost(body, authService))
    }

    private async buildHttpPost(body: any, authService?: AuthService): Promise<RequestInit> {
        let headers = { "Content-Type": "application/json" } as any
        if (authService) {
            let user = await authService.getUser()
            headers['Authorization'] = "Bearer " + user!.access_token
        }
        return {
            method: "POST",
            headers: headers,
            body: JSON.stringify(body)
        }
    }

    public activatesOn(req: browser.WebRequest.OnCompletedDetailsType): boolean {
        if (!this.activationRule) {
            return false
        }
        return this.requestMatchesRuleCondition(req, this.activationRule)
    }

    private requestMatchesRuleCondition(req: browser.WebRequest.OnCompletedDetailsType, rule: AgentRule): boolean {
        return new RegExp(rule.condition.urlRegex!).test(req.url)
            && (!rule.condition.requestMethods || rule.condition.requestMethods!.includes(req.method.toLowerCase()))
            && (!rule.condition.resourceTypes || rule.condition.resourceTypes!.includes(req.type))
    }

    public findMatchingActions(req: browser.WebRequest.OnCompletedDetailsType): AgentRuleAction[] {
        return this.manifest.onHttpRequest ? this.manifest.onHttpRequest.filter(r => this.requestMatchesRuleCondition(req, r))
            .flatMap(r => r.actions) : []
    }

    public async * ask(msg: string, sessionId: string, authService?: AuthService): AsyncIterable<string> {
        let ret = await fetchStreamJson(`${this.sessionUrl(sessionId)}/questions`, await this.buildHttpPost({ question: msg }, authService))
        for await (const part of ret) {
            if (typeof part === "string") {
                yield part
            } else {
                yield part.answer
            }
        }
    }

    private sessionUrl(sessionId: string): string {
        return `${this.url}/sessions/${sessionId}`
    }

    public async transcriptAudio(audioFileBase64: string, sessionId: string, authService?: AuthService) {
        let ret = await this.postJson(`${this.sessionUrl(sessionId)}/transcriptions`, { file: audioFileBase64 }, authService)
        return ret.text
    }

    public async solveInteractionSummary(detail: any, sessionId: string, authService?: AuthService): Promise<string> {
        let ret = await this.postJson(`${this.sessionUrl(sessionId)}/interactions`, detail, authService)
        return ret.summary
    }

}

export interface AgentManifest {
    id: string
    name: string
    capabilities?: string[]
    welcomeMessage: string
    prompts?: Prompt[]
    onSessionClose?: EndAction
    onHttpRequest?: AgentRule[]
    auth?: AuthConfig
    contactEmail: string
}

export interface AgentRule {
    condition: AgentRuleCondition
    actions: AgentRuleAction[]
}

export interface AgentRuleCondition {
    urlRegex: string
    requestMethods?: string[]
    resourceTypes?: string[]
}

export interface AgentRuleAction {
    activate?: ActivationAction
    addHeader?: AddHeaderRuleAction
    recordInteraction?: RecordInteractionRuleAction
}

export interface ActivationAction {
    httpRequest?: HttpRequestAction
}

export interface EndAction {
    httpRequest: HttpRequestAction
}

export interface HttpRequestAction {
    url: string
    method?: string
}

export interface AddHeaderRuleAction {
    header: string
    value: string
}

export interface RecordInteractionRuleAction {
    url: string
}

export interface AgentSession {
    id: string
}
