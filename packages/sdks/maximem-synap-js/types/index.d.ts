export type LogLevel = 'debug' | 'error';
export type RetrievalMode = 'fast' | 'accurate';
export type IngestMode = 'fast' | 'long-range';
export type PromptStyle = 'structured' | 'narrative' | 'bullet_points';
export type ContextType =
  | 'all'
  | 'facts'
  | 'preferences'
  | 'episodes'
  | 'emotions'
  | 'temporal_events';
export type FlattenedContextType =
  | 'fact'
  | 'preference'
  | 'episode'
  | 'emotion'
  | 'temporal_event';
export type DocumentType =
  | 'ai-chat-conversation'
  | 'document'
  | 'email'
  | 'pdf'
  | 'image'
  | 'audio'
  | 'meeting-transcript';

export interface BridgeLogHandler {
  (level: LogLevel, message: string): void;
}

export interface SynapClientOptions {
  apiKey?: string;
  baseUrl?: string;
  grpcHost?: string;
  grpcPort?: number;
  grpcUseTls?: boolean;
  pythonBin?: string;
  bridgeScriptPath?: string;
  sdkHome?: string;
  venvPath?: string;
  pythonBootstrap?: string;
  pythonPackage?: string;
  pythonSdkVersion?: string;
  noDeps?: boolean;
  noBuildIsolation?: boolean;
  upgrade?: boolean;
  forceRecreateVenv?: boolean;
  autoSetup?: boolean;
  requestTimeoutMs?: number;
  initTimeoutMs?: number;
  ingestTimeoutMs?: number;
  pythonEnv?: NodeJS.ProcessEnv;
  onLog?: BridgeLogHandler;
}

export interface ChatMessage {
  role?: 'user' | 'assistant';
  content: string;
  metadata?: Record<string, string>;
}

export interface AddMemoryInput {
  userId: string;
  customerId: string;
  conversationId?: string;
  sessionId?: string;
  messages: ChatMessage[];
  mode?: IngestMode;
  documentType?: DocumentType;
  documentId?: string;
  documentCreatedAt?: string;
  metadata?: Record<string, unknown>;
}

export interface SearchMemoryInput {
  userId: string;
  customerId?: string;
  query: string;
  maxResults?: number;
  mode?: RetrievalMode;
  conversationId?: string;
  types?: ContextType[];
}

export interface GetMemoriesInput {
  userId: string;
  customerId?: string;
  mode?: RetrievalMode;
  conversationId?: string;
  maxResults?: number;
  types?: ContextType[];
}

export interface DeleteMemoryInput {
  userId: string;
  customerId?: string;
  memoryId?: string | null;
}

export interface FetchUserContextInput {
  userId: string;
  customerId?: string;
  conversationId?: string;
  searchQuery?: string[];
  maxResults?: number;
  types?: ContextType[];
  mode?: RetrievalMode;
}

export interface FetchCustomerContextInput {
  customerId: string;
  conversationId?: string;
  searchQuery?: string[];
  maxResults?: number;
  types?: ContextType[];
  mode?: RetrievalMode;
}

export interface FetchClientContextInput {
  conversationId?: string;
  searchQuery?: string[];
  maxResults?: number;
  types?: ContextType[];
  mode?: RetrievalMode;
}

export interface GetContextForPromptInput {
  conversationId: string;
  style?: PromptStyle;
}

export interface BridgeStepTiming {
  step: string;
  ms: number;
}

export interface BridgeTiming {
  python_total_ms: number;
  steps: BridgeStepTiming[];
}

export interface TemporalFields {
  eventDate: string | null;
  validUntil: string | null;
  temporalCategory: string | null;
  temporalConfidence: number;
}

export interface Fact extends TemporalFields {
  id: string;
  content: string;
  confidence: number;
  source: string;
  extractedAt: string | null;
  metadata: Record<string, unknown>;
}

export interface Preference extends TemporalFields {
  id: string;
  category: string;
  content: string;
  strength: number;
  source: string;
  extractedAt: string | null;
  metadata: Record<string, unknown>;
}

export interface Episode extends TemporalFields {
  id: string;
  summary: string;
  occurredAt: string | null;
  significance: number;
  participants: string[];
  metadata: Record<string, unknown>;
}

export interface Emotion extends TemporalFields {
  id: string;
  emotionType: string;
  intensity: number;
  detectedAt: string | null;
  context: string;
  metadata: Record<string, unknown>;
}

export interface TemporalEvent {
  id: string;
  content: string;
  eventDate: string | null;
  validUntil: string | null;
  temporalCategory: string;
  temporalConfidence: number;
  confidence: number;
  source: string;
  extractedAt: string | null;
  metadata: Record<string, unknown>;
}

export interface RecentMessage {
  role: string;
  content: string;
  timestamp: string | null;
  messageId: string;
}

export interface ContextResponseMetadata {
  correlationId: string;
  ttlSeconds: number;
  source: string;
  retrievedAt: string | null;
  compactionApplied: string | null;
}

export interface ConversationContext {
  summary: string | null;
  currentState: Record<string, unknown>;
  keyExtractions: Record<string, Array<Record<string, unknown>>>;
  recentTurns: Array<Record<string, unknown>>;
  compactionId: string | null;
  compactedAt: string | null;
  conversationId: string | null;
}

export interface ContextResponse {
  facts: Fact[];
  preferences: Preference[];
  episodes: Episode[];
  emotions: Emotion[];
  temporalEvents: TemporalEvent[];
  conversationContext: ConversationContext | null;
  metadata: ContextResponseMetadata;
  rawResponse: Record<string, unknown>;
  bridgeTiming?: BridgeTiming;
}

export interface ContextForPromptResult {
  formattedContext: string | null;
  available: boolean;
  isStale: boolean;
  compressionRatio: number | null;
  validationScore: number | null;
  compactionAgeSeconds: number | null;
  qualityWarning: boolean;
  recentMessages: RecentMessage[];
  recentMessageCount: number;
  compactedMessageCount: number;
  totalMessageCount: number;
  bridgeTiming?: BridgeTiming;
}

export interface AddMemoryResult {
  success: boolean;
  latencyMs: number;
  rawResponse: Record<string, unknown>;
  note?: string;
  bridgeTiming?: BridgeTiming;
}

export interface SearchMemoryItem extends TemporalFields {
  id: string;
  memory: string;
  score?: number;
  source?: string;
  metadata: Record<string, unknown>;
  contextType?: FlattenedContextType;
}

export interface SearchMemoryResult {
  success: boolean;
  latencyMs: number;
  results: SearchMemoryItem[];
  resultsCount: number;
  rawResponse: Record<string, unknown>;
  source?: string;
  bridgeTiming?: BridgeTiming;
}

export interface MemoryItem extends SearchMemoryItem {}

export interface GetMemoriesResult {
  success: boolean;
  latencyMs: number;
  memories: MemoryItem[];
  totalCount: number;
  rawResponse: Record<string, unknown> | null;
  source?: string;
  bridgeTiming?: BridgeTiming;
}

export interface DeleteMemoryResult {
  success: boolean;
  latencyMs: number;
  deletedCount: number;
  rawResponse: Record<string, unknown> | null;
  note?: string;
  bridgeTiming?: BridgeTiming;
}

export class SynapClient {
  constructor(options?: SynapClientOptions);
  init(): Promise<void>;
  addMemory(input: AddMemoryInput): Promise<AddMemoryResult>;
  searchMemory(input: SearchMemoryInput): Promise<SearchMemoryResult>;
  getMemories(input: GetMemoriesInput): Promise<GetMemoriesResult>;
  fetchUserContext(input: FetchUserContextInput): Promise<ContextResponse>;
  fetchCustomerContext(input: FetchCustomerContextInput): Promise<ContextResponse>;
  fetchClientContext(input?: FetchClientContextInput): Promise<ContextResponse>;
  getContextForPrompt(input: GetContextForPromptInput): Promise<ContextForPromptResult>;
  deleteMemory(input: DeleteMemoryInput): Promise<DeleteMemoryResult>;
  shutdown(): Promise<void>;
}

export interface SetupPythonRuntimeOptions {
  sdkHome?: string;
  venvPath?: string;
  pythonBootstrap?: string;
  pythonPackage?: string;
  pythonSdkVersion?: string;
  noDeps?: boolean;
  noBuildIsolation?: boolean;
  upgrade?: boolean;
  forceRecreateVenv?: boolean;
}

export interface SetupPythonRuntimeResult {
  sdkHome: string;
  venvPath: string;
  pythonBin: string;
  pythonPackage: string;
  pythonSdkVersion: string | null;
  installTarget: string;
}

export interface SetupTypeScriptExtensionOptions {
  projectDir?: string;
  packageManager?: 'npm' | 'pnpm' | 'yarn' | 'bun';
  skipInstall?: boolean;
  tsconfigPath?: string;
  wrapperPath?: string;
  force?: boolean;
  noWrapper?: boolean;
}

export interface SetupTypeScriptExtensionResult {
  projectDir: string;
  packageManager: 'npm' | 'pnpm' | 'yarn' | 'bun';
  installedDevDependencies: boolean;
  tsconfigPath: string;
  tsconfigCreated: boolean;
  wrapperPath: string | null;
  wrapperCreated: boolean;
}

// Error classes
export class SynapError extends Error {
  correlationId: string | null;
  constructor(message: string, correlationId?: string);
}
export class SynapTransientError extends SynapError {}
export class SynapPermanentError extends SynapError {}
export class NetworkTimeoutError extends SynapTransientError {}
export class RateLimitError extends SynapTransientError {
  retryAfterSeconds: number | null;
  constructor(message: string, retryAfterSeconds?: number, correlationId?: string);
}
export class ServiceUnavailableError extends SynapTransientError {}
export class AgentUnavailableError extends SynapTransientError {}
export class InvalidInputError extends SynapPermanentError {}
export class InvalidInstanceIdError extends InvalidInputError {}
export class InvalidConversationIdError extends InvalidInputError {}
export class AuthenticationError extends SynapPermanentError {}
export class InsufficientCreditsError extends SynapPermanentError {
  balanceCredits: number | null;
  minimumRequiredCredits: number | null;
  recoveryUrl: string | null;
  redeemUrl: string | null;
  constructor(
    message: string,
    details?: {
      balance_credits?: number | null;
      minimum_required_credits?: number | null;
      recovery_url?: string | null;
      redeem_url?: string | null;
    },
    correlationId?: string,
  );
}
export class ContextNotFoundError extends SynapPermanentError {}
export class SessionExpiredError extends SynapPermanentError {}
export class ListeningAlreadyActiveError extends SynapPermanentError {}
export class ListeningNotActiveError extends SynapPermanentError {}

export function createClient(options?: SynapClientOptions): SynapClient;
export function resolveInstanceId(explicitInstanceId?: string): string;
export function setupPythonRuntime(
  options?: SetupPythonRuntimeOptions
): Promise<SetupPythonRuntimeResult>;
export function setupTypeScriptExtension(
  options?: SetupTypeScriptExtensionOptions
): Promise<SetupTypeScriptExtensionResult>;
