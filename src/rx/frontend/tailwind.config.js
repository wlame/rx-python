/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}', './index.html'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // GitHub-inspired color palette
        'gh-canvas': {
          default: '#ffffff',
          subtle: '#f6f8fa',
          inset: '#eff2f5',
        },
        'gh-canvas-dark': {
          default: '#0d1117',
          subtle: '#161b22',
          inset: '#010409',
        },
        'gh-border': {
          default: '#d0d7de',
          muted: '#d8dee4',
        },
        'gh-border-dark': {
          default: '#30363d',
          muted: '#21262d',
        },
        'gh-fg': {
          default: '#1f2328',
          muted: '#656d76',
          subtle: '#6e7781',
        },
        'gh-fg-dark': {
          default: '#e6edf3',
          muted: '#8b949e',
          subtle: '#6e7681',
        },
        'gh-accent': {
          fg: '#0969da',
          emphasis: '#0969da',
          muted: 'rgba(84, 174, 255, 0.4)',
        },
        'gh-accent-dark': {
          fg: '#58a6ff',
          emphasis: '#1f6feb',
          muted: 'rgba(56, 139, 253, 0.4)',
        },
        'gh-success': {
          fg: '#1a7f37',
          emphasis: '#1f883d',
        },
        'gh-success-dark': {
          fg: '#3fb950',
          emphasis: '#238636',
        },
        'gh-attention': {
          fg: '#9a6700',
          emphasis: '#bf8700',
        },
        'gh-attention-dark': {
          fg: '#d29922',
          emphasis: '#9e6a03',
        },
        'gh-danger': {
          fg: '#d1242f',
          emphasis: '#cf222e',
        },
        'gh-danger-dark': {
          fg: '#f85149',
          emphasis: '#da3633',
        },
      },
      fontFamily: {
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'SF Mono',
          'Menlo',
          'Consolas',
          'Liberation Mono',
          'monospace',
        ],
      },
      fontSize: {
        code: ['12px', '20px'],
      },
    },
  },
  plugins: [
    function ({ addUtilities }) {
      addUtilities({
        '.scrollbar-hide': {
          /* Firefox */
          'scrollbar-width': 'none',
          /* Safari and Chrome */
          '&::-webkit-scrollbar': {
            display: 'none'
          }
        }
      })
    }
  ],
};
