package com.cnnspeed.app.ekf

/**
 * Minimal dense float matrix for EKF math. Row-major.
 * Sized for our state vector (6) and small measurement vectors (≤6).
 */
class FloatMatrix(val rows: Int, val cols: Int, val data: FloatArray) {

    constructor(rows: Int, cols: Int) : this(rows, cols, FloatArray(rows * cols))

    operator fun get(r: Int, c: Int): Float = data[r * cols + c]
    operator fun set(r: Int, c: Int, v: Float) { data[r * cols + c] = v }

    operator fun plus(o: FloatMatrix): FloatMatrix {
        require(rows == o.rows && cols == o.cols)
        val out = FloatMatrix(rows, cols)
        for (i in data.indices) out.data[i] = data[i] + o.data[i]
        return out
    }

    operator fun minus(o: FloatMatrix): FloatMatrix {
        require(rows == o.rows && cols == o.cols)
        val out = FloatMatrix(rows, cols)
        for (i in data.indices) out.data[i] = data[i] - o.data[i]
        return out
    }

    operator fun times(s: Float): FloatMatrix {
        val out = FloatMatrix(rows, cols)
        for (i in data.indices) out.data[i] = data[i] * s
        return out
    }

    operator fun times(o: FloatMatrix): FloatMatrix {
        require(cols == o.rows) { "Matmul shape mismatch ${rows}x${cols} * ${o.rows}x${o.cols}" }
        val out = FloatMatrix(rows, o.cols)
        for (i in 0 until rows) {
            for (k in 0 until cols) {
                val a = this[i, k]
                if (a == 0f) continue
                for (j in 0 until o.cols) {
                    out[i, j] = out[i, j] + a * o[k, j]
                }
            }
        }
        return out
    }

    fun transpose(): FloatMatrix {
        val out = FloatMatrix(cols, rows)
        for (i in 0 until rows) for (j in 0 until cols) out[j, i] = this[i, j]
        return out
    }

    /** In-place add to diagonal (used for ε regularization). */
    fun addDiag(v: Float): FloatMatrix {
        val n = minOf(rows, cols)
        for (i in 0 until n) this[i, i] = this[i, i] + v
        return this
    }

    fun copy(): FloatMatrix = FloatMatrix(rows, cols, data.copyOf())

    /** Inverse via Gauss-Jordan with partial pivoting. Returns null if singular. */
    fun inverse(): FloatMatrix? {
        require(rows == cols)
        val n = rows
        val a = Array(n) { i -> FloatArray(2 * n).also { row ->
            for (j in 0 until n) row[j] = this[i, j]
            row[n + i] = 1f
        } }
        for (i in 0 until n) {
            var pivot = i
            var maxAbs = kotlin.math.abs(a[i][i])
            for (r in i + 1 until n) {
                val v = kotlin.math.abs(a[r][i])
                if (v > maxAbs) { maxAbs = v; pivot = r }
            }
            if (maxAbs < 1e-12f) return null
            if (pivot != i) { val tmp = a[i]; a[i] = a[pivot]; a[pivot] = tmp }
            val piv = a[i][i]
            for (j in 0 until 2 * n) a[i][j] /= piv
            for (r in 0 until n) {
                if (r == i) continue
                val f = a[r][i]
                if (f == 0f) continue
                for (j in 0 until 2 * n) a[r][j] -= f * a[i][j]
            }
        }
        val out = FloatMatrix(n, n)
        for (i in 0 until n) for (j in 0 until n) out[i, j] = a[i][n + j]
        return out
    }

    fun diagMean(): Float {
        val n = minOf(rows, cols)
        if (n == 0) return 0f
        var s = 0f
        for (i in 0 until n) s += this[i, i]
        return s / n
    }

    companion object {
        fun identity(n: Int): FloatMatrix {
            val m = FloatMatrix(n, n)
            for (i in 0 until n) m[i, i] = 1f
            return m
        }
        fun column(vararg v: Float): FloatMatrix {
            val m = FloatMatrix(v.size, 1)
            for (i in v.indices) m[i, 0] = v[i]
            return m
        }
        fun diag(vararg v: Float): FloatMatrix {
            val m = FloatMatrix(v.size, v.size)
            for (i in v.indices) m[i, i] = v[i]
            return m
        }
    }
}
