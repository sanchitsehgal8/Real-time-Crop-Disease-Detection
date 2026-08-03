import { defineCloudflareConfig } from "@opennextjs/cloudflare"

// No incremental-cache override: the app is a single client-rendered page with
// no ISR or on-demand revalidation, so the in-worker default is sufficient.
// If ISR is added later, wire up an R2 or KV cache here and add the matching
// binding to wrangler.jsonc.
export default defineCloudflareConfig()
