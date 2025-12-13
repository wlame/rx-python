import { writable } from 'svelte/store';

export interface Notification {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  message: string;
  duration?: number; // milliseconds, undefined = manual dismiss
}

function createNotificationsStore() {
  const { subscribe, update } = writable<Notification[]>([]);

  function show(type: Notification['type'], message: string, duration = 5000) {
    const id = `${Date.now()}-${Math.random()}`;
    const notification: Notification = { id, type, message, duration };

    update((notifications) => [...notifications, notification]);

    if (duration) {
      setTimeout(() => {
        dismiss(id);
      }, duration);
    }

    return id;
  }

  function dismiss(id: string) {
    update((notifications) => notifications.filter((n) => n.id !== id));
  }

  function error(message: string, duration?: number) {
    return show('error', message, duration);
  }

  function warning(message: string, duration?: number) {
    return show('warning', message, duration);
  }

  function info(message: string, duration?: number) {
    return show('info', message, duration);
  }

  function success(message: string, duration?: number) {
    return show('success', message, duration);
  }

  return {
    subscribe,
    show,
    dismiss,
    error,
    warning,
    info,
    success,
  };
}

export const notifications = createNotificationsStore();
