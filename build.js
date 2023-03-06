/* eslint-disable @typescript-eslint/no-var-requires */
const child_process = require('child_process')

const esbuild = require('esbuild')
const { copy } = require('esbuild-plugin-copy')
const svgrPlugin = require('esbuild-plugin-svgr')

const production = process.env.NODE_ENV === 'production'

const tailwindCSSPlugin = {
  name: 'tailwindcss',
  setup: (build) => {
    build.onEnd(() => {
      const { stdout, stderr } = child_process.spawnSync('tailwindcss', [
        '--input',
        './public/main.css',
        '--output',
        './dist/main.css',
        '--color',
        production ? '--minify' : '',
      ])

      process.stdout.write(stdout)
      process.stderr.write(stderr)
    })
  },
}

const watch = ['y', 'yes', 'true', '1'].includes(
  process.env.BUILD_WATCH?.toLowerCase()
)

const opts = {
  entryPoints: ['src/index.tsx'],
  bundle: true,
  drop: production ? ['console', 'debugger'] : [],
  minify: production,
  sourcemap: production ? false : 'inline',
  plugins: [
    tailwindCSSPlugin,
    svgrPlugin({ icon: true }),
    copy({
      resolveFrom: 'cwd',
      assets: {
        from: ['./public/index.html'],
        to: ['./dist/index.html'],
      },
    }),
  ],
  outdir: 'dist',
  logLevel: 'info',
}

;(async () => {
  if (watch) {
    const ctx = await esbuild.context(opts)
    await ctx.watch()
  } else {
    await esbuild.build(opts)
  }
})()
