export function useBlogSupabase() {
  const app = useNuxtApp()
  return {
    configured: app.$supabaseConfigured,
    supabase: app.$supabase,
  }
}
