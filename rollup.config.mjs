import commonjs from '@rollup/plugin-commonjs'
import { nodeResolve } from '@rollup/plugin-node-resolve'
import replace from '@rollup/plugin-replace'
import terser from '@rollup/plugin-terser'
import typescript from '@rollup/plugin-typescript'
import autoprefixer from 'autoprefixer'
import copy from 'rollup-plugin-copy'
import livereload from 'rollup-plugin-livereload'
import postcss from 'rollup-plugin-postcss'
import serve from 'rollup-plugin-serve'
import wasmImport from 'rollup-wasm-pack-import'
import tailwindcss from 'tailwindcss'

import wasmPack from './rollup-plugin-wasm-pack.mjs'

/**
 * @type {import("rollup").RollupOptions}
 */
export default {
  input: 'src/index.tsx',
  output: {
    dir: 'dist',
    format: 'iife',
    inlineDynamicImports: true,
    sourcemap: process.env.NODE_ENV === 'development',
  },
  plugins: [
    wasmPack({
      packages: {
        'rps-network': './rps-network',
      },
    }),
    typescript(),
    nodeResolve(),
    commonjs(),
    replace({
      'process.env.NODE_ENV': JSON.stringify(
        process.env.NODE_ENV ?? 'production'
      ),
      preventAssignment: true,
    }),
    wasmImport({
      copy: true,
      serverPath: '/',
      mapping: {
        'rps-network': 'rps_network_bg.wasm',
      },
    }),
    copy({
      targets: [{ src: 'public/index.html', dest: 'dist/' }],
    }),
    postcss({ plugins: [tailwindcss, autoprefixer] }),
    process.env.NODE_ENV === 'production' ? terser() : null,
    process.env.ROLLUP_WATCH === 'true'
      ? serve({ contentBase: './dist' })
      : null,
    process.env.ROLLUP_WATCH === 'true'
      ? livereload({ watch: './dist' })
      : null,
  ],
}
