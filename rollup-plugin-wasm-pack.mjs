import child_process from 'child_process'
import path from 'path'

/**
 * @type {() => import("rollup").Plugin}
 */
export default function ({ packages }) {
  const builtPackages = {}

  return {
    name: 'wasm-pack',
    buildStart() {
      for (const [pkg, pkgPath] of Object.entries(packages)) {
        const built = !!builtPackages[pkg]

        const resolvedPkgPath = path.resolve(pkgPath)
        const files = [
          path.join(resolvedPkgPath, 'Cargo.toml'),
          path.join(resolvedPkgPath, 'src'),
        ]
        builtPackages[pkg] = files

        for (const f of files) {
          this.addWatchFile(f)
        }

        if (built) {
          continue
        }

        console.log(`compiling ${pkg}`)
        const out = child_process.spawnSync(
          'wasm-pack',
          ['build', '--target', 'web'],
          {
            cwd: resolvedPkgPath,
          }
        )
        if (out.error) {
          this.error(out.error)
        }
        console.log(out.stderr.toString('utf-8'))
      }
    },
    watchChange(id) {
      for (const [pkg, files] of Object.entries(builtPackages)) {
        for (const f of files) {
          if (path.resolve(id).startsWith(path.resolve(f))) {
            builtPackages[pkg] = false
            break
          }
        }
      }
    },
  }
}
