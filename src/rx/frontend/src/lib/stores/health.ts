import { writable } from 'svelte/store';
import { api } from '../api';
import type { HealthResponse } from '../types';

interface HealthState {
  connected: boolean;
  loading: boolean;
  error: string | null;
  data: HealthResponse | null;
}

function createHealthStore() {
  const { subscribe, set, update } = writable<HealthState>({
    connected: false,
    loading: true,
    error: null,
    data: null,
  });

  let checkInterval: ReturnType<typeof setInterval> | null = null;

  async function check() {
    update((s) => ({ ...s, loading: true }));
    try {
      const data = await api.getHealth();
      set({ connected: true, loading: false, error: null, data });
    } catch (e) {
      set({
        connected: false,
        loading: false,
        error: e instanceof Error ? e.message : 'Connection failed',
        data: null,
      });
    }
  }

  function startPolling(intervalMs: number = 300000) {
    stopPolling();
    check();
    checkInterval = setInterval(check, intervalMs);
  }

  function stopPolling() {
    if (checkInterval) {
      clearInterval(checkInterval);
      checkInterval = null;
    }
  }

  return {
    subscribe,
    check,
    startPolling,
    stopPolling,
  };
}

export const health = createHealthStore();
