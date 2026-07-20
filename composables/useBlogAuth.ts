import type { Session } from '@supabase/supabase-js'

let authListenerBound = false

export function useBlogAuth() {
  const { configured, supabase } = useBlogSupabase()
  const session = useState<Session | null>('blog-auth-session', () => null)
  const ready = useState('blog-auth-ready', () => false)

  async function initialize() {
    if (!import.meta.client || ready.value) return
    if (!configured || !supabase) {
      ready.value = true
      return
    }

    const { data, error } = await supabase.auth.getSession()
    if (error) throw error
    session.value = data.session
    ready.value = true

    if (!authListenerBound) {
      supabase.auth.onAuthStateChange((_event, nextSession) => {
        session.value = nextSession
        ready.value = true
      })
      authListenerBound = true
    }
  }

  async function signOut() {
    if (!supabase) return
    const { error } = await supabase.auth.signOut()
    if (error) throw error
    session.value = null
  }

  return {
    configured,
    initialize,
    ready,
    session,
    signOut,
    supabase,
    user: computed(() => session.value?.user ?? null),
  }
}
