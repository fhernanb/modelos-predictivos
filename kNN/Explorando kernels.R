# -------------------------------------------------------------------------
# En este script vamos a explorar las formas de algunos de los
# kernel usados en el paquete kknn
# -------------------------------------------------------------------------

# Definiendo los kernel
rect_ker <- function(d) (abs(d) <= 1) / 2
epan_ker <- function(d) 0.75 * (1-d^2) * (abs(d) <= 1)

# Dibujando las curvas
curve(expr=rect_ker, from=0, to=3, las=1, lwd=4, col='blue',
      ylim=c(0, 1), ylab="K(d)", xlab="Distancia (d)")
curve(expr=epan_ker, add=TRUE, lwd=2, col="orange")

# Tarea:
# 1) Dibujar los otros kernel.
# 2) Por que solo se pinta el kernel en el primer cuadrante?
