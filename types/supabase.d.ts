import type { SupabaseClient } from '@supabase/supabase-js'

declare module '#app' {
  interface NuxtApp {
    $supabase: SupabaseClient | null
    $supabaseConfigured: boolean
  }
}

declare module 'vue' {
  interface ComponentCustomProperties {
    $supabase: SupabaseClient | null
    $supabaseConfigured: boolean
  }
}

export {}
