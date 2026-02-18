import { createClient } from '@supabase/supabase-js';

/**
 * Supabase server client with service role key
 * IMPORTANT: Only use this in server-side code (API routes, server components)
 * Never expose SUPABASE_SERVICE_ROLE_KEY to the browser
 */

const supabaseUrl = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceRoleKey) {
  throw new Error(
    'Missing required Supabase server environment variables. ' +
    'Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY'
  );
}

export const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false,
    detectSessionInUrl: false,
  },
});

export type SupabaseAdminClient = typeof supabaseAdmin;
