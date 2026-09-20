import { useEffect, useState } from "react";
import type { User } from "@supabase/supabase-js";
import { supabase } from "../auths/supabaseClients.ts";

export { supabase };

// Initiate Google OAuth directly without needing to navigate to another page first
export async function signInWithGoogle(redirectTo?: string) {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: {
      redirectTo: redirectTo || window.location.origin,
    },
  });

  if (error) {
    console.error("Error logging in with Google:", error.message);
  }

  return { data, error };
}

// Sign out from anywhere in the app
export async function signOut() {
  const { error } = await supabase.auth.signOut();
  if (error) {
    console.error("Error signing out:", error.message);
  }
  return { error };
}

export function Login() {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // 1. Check current session on component mount
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
    });

    // 2. Listen for auth changes (login, logout, token refresh)
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  // Handler to initiate Google OAuth login
  const handleGoogleLogin = async () => {
    await signInWithGoogle();
  };

  // Handler to log out
  const handleSignOut = async () => {
    await signOut();
  };

  return (
    <div style={{ padding: "40px", fontFamily: "sans-serif", textAlign: "center" }}>
      {user ? (
        <div>
          <h2>Welcome, {user.user_metadata?.full_name || user.email}!</h2>
          {user.user_metadata?.avatar_url && (
            <img
              src={user.user_metadata.avatar_url}
              alt="Profile"
              style={{ borderRadius: "50%", width: "80px", height: "80px", margin: "10px 0" }}
            />
          )}
          <p>Email: {user.email}</p>
          <button onClick={handleSignOut} style={{ padding: "10px 20px", cursor: "pointer" }}>
            Sign Out
          </button>
        </div>
      ) : (
        <div>
          <h2>Sign In to Your App</h2>
          <button
            onClick={handleGoogleLogin}
            style={{
              padding: "12px 24px",
              fontSize: "16px",
              cursor: "pointer",
              backgroundColor: "#4285F4",
              color: "#fff",
              border: "none",
              borderRadius: "4px",
              fontWeight: "bold",
            }}
          >
            Sign in with Google
          </button>
        </div>
      )}
    </div>
  );
}