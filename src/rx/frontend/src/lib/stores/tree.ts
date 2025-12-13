import { writable, get } from 'svelte/store';
import { api } from '../api';
import type { TreeNode, TreeEntry } from '../types';

interface TreeState {
  roots: TreeNode[];
  loading: boolean;
  error: string | null;
  selectedPath: string | null;
}

function entryToNode(entry: TreeEntry, level: number): TreeNode {
  return {
    ...entry,
    expanded: false,
    loading: false,
    children: [],
    level,
  };
}

function createTreeStore() {
  const { subscribe, set, update } = writable<TreeState>({
    roots: [],
    loading: false,
    error: null,
    selectedPath: null,
  });

  function findNode(
    nodes: TreeNode[],
    path: string
  ): TreeNode | null {
    for (const node of nodes) {
      if (node.path === path) return node;
      if (node.children.length > 0) {
        const found = findNode(node.children, path);
        if (found) return found;
      }
    }
    return null;
  }

  function updateNode(
    nodes: TreeNode[],
    path: string,
    updater: (node: TreeNode) => TreeNode
  ): TreeNode[] {
    return nodes.map((node) => {
      if (node.path === path) {
        return updater(node);
      }
      if (node.children.length > 0) {
        return {
          ...node,
          children: updateNode(node.children, path, updater),
        };
      }
      return node;
    });
  }

  async function loadRoots() {
    update((s) => ({ ...s, loading: true, error: null }));
    try {
      const response = await api.getTree();
      const roots = response.entries
        .filter((e) => e.type === 'directory')
        .map((e) => entryToNode(e, 0));
      update((s) => ({ ...s, roots, loading: false }));
    } catch (e) {
      update((s) => ({
        ...s,
        loading: false,
        error: e instanceof Error ? e.message : 'Failed to load roots',
      }));
    }
  }

  async function loadDirectory(path: string) {
    const state = get({ subscribe });
    const node = findNode(state.roots, path);
    if (!node || node.type !== 'directory') return;

    update((s) => ({
      ...s,
      roots: updateNode(s.roots, path, (n) => ({ ...n, loading: true })),
    }));

    try {
      const response = await api.getTree(path);
      const children = response.entries.map((e) =>
        entryToNode(e, node.level + 1)
      );
      update((s) => ({
        ...s,
        roots: updateNode(s.roots, path, (n) => ({
          ...n,
          loading: false,
          children,
          expanded: true,
        })),
      }));
    } catch (e) {
      update((s) => ({
        ...s,
        roots: updateNode(s.roots, path, (n) => ({ ...n, loading: false })),
        error: e instanceof Error ? e.message : 'Failed to load directory',
      }));
    }
  }

  async function toggleExpanded(path: string) {
    const state = get({ subscribe });
    const node = findNode(state.roots, path);
    if (!node || node.type !== 'directory') return;

    if (node.expanded) {
      // Collapse
      update((s) => ({
        ...s,
        roots: updateNode(s.roots, path, (n) => ({ ...n, expanded: false })),
      }));
    } else if (node.children.length > 0) {
      // Already loaded, just expand
      update((s) => ({
        ...s,
        roots: updateNode(s.roots, path, (n) => ({ ...n, expanded: true })),
      }));
    } else {
      // Load and expand
      await loadDirectory(path);
    }
  }

  function selectPath(path: string | null) {
    update((s) => ({ ...s, selectedPath: path }));
  }

  function clearError() {
    update((s) => ({ ...s, error: null }));
  }

  /**
   * Expand all parent directories to reveal a file path and select it
   */
  async function expandToPath(filePath: string) {
    // Ensure roots are loaded first
    const state = get({ subscribe });
    if (state.roots.length === 0 && !state.loading) {
      await loadRoots();
    }

    // Wait for roots to finish loading if they're currently loading
    if (state.loading) {
      // Poll until loading is done
      await new Promise<void>((resolve) => {
        const unsubscribe = subscribe((s) => {
          if (!s.loading) {
            unsubscribe();
            resolve();
          }
        });
      });
    }

    // Get all parent directory paths
    const parts = filePath.split('/').filter(p => p);
    const parentPaths: string[] = [];

    for (let i = 1; i < parts.length; i++) {
      parentPaths.push('/' + parts.slice(0, i).join('/'));
    }

    // Load and expand each parent directory in sequence
    for (const dirPath of parentPaths) {
      const currentState = get({ subscribe });
      const node = findNode(currentState.roots, dirPath);

      if (node && node.type === 'directory' && !node.expanded) {
        await loadDirectory(dirPath);
      }
    }

    // Select the file
    selectPath(filePath);
  }

  return {
    subscribe,
    loadRoots,
    loadDirectory,
    toggleExpanded,
    selectPath,
    clearError,
    expandToPath,
  };
}

export const tree = createTreeStore();
