import { writable, derived } from 'svelte/store';
import type { AppSettings, Theme } from '../types';

const STORAGE_KEY = 'rx-settings';

const defaultSettings: AppSettings = {
  theme: 'system',
  sidebarWidth: 280,
  editorFontSize: 13,
  showLineNumbers: true,
  wrapLines: false,
  linesPerPage: 1000,
};

function loadSettings(): AppSettings {
  if (typeof localStorage === 'undefined') return defaultSettings;
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return defaultSettings;
  try {
    return { ...defaultSettings, ...JSON.parse(stored) };
  } catch {
    return defaultSettings;
  }
}

function createSettingsStore() {
  const { subscribe, set, update } = writable<AppSettings>(loadSettings());

  return {
    subscribe,
    set(value: AppSettings) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
      set(value);
    },
    update(updater: (settings: AppSettings) => AppSettings) {
      update((current) => {
        const updated = updater(current);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        return updated;
      });
    },
    reset() {
      localStorage.removeItem(STORAGE_KEY);
      set(defaultSettings);
    },
  };
}

export const settings = createSettingsStore();

// Derived store for actual theme (resolves 'system' to 'light' or 'dark')
export const resolvedTheme = derived(settings, ($settings): Theme => {
  if ($settings.theme === 'system') {
    if (typeof window === 'undefined') return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }
  return $settings.theme;
});

// Apply theme to document
if (typeof window !== 'undefined') {
  resolvedTheme.subscribe((theme) => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  });

  // Listen for system theme changes
  window
    .matchMedia('(prefers-color-scheme: dark)')
    .addEventListener('change', () => {
      settings.update((s) => ({ ...s })); // Trigger re-evaluation
    });
}
