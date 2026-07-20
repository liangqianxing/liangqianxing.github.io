import { createClient, type SupabaseClient } from '@supabase/supabase-js'

export default defineNuxtPlugin(() => {
  const config = useRuntimeConfig()
  const url = String(config.public.supabaseUrl ?? '')
  const anonKey = String(config.public.supabaseAnonKey ?? '')
  const configured = Boolean(url && anonKey)

  let supabase: SupabaseClient | null = null
  if (configured && import.meta.client) {
    supabase = createClient(url, anonKey, {
      auth: {
        detectSessionInUrl: false,
        flowType: 'pkce',
        persistSession: true,
      },
    })
  }

  return {
    provide: {
      supabase,
      supabaseConfigured: configured,
    },
  }
})
