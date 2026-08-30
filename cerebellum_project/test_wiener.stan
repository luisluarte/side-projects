data {
  array[10] real y;
}
parameters {
  real a;
  real t0;
  real w;
  real v;
}
model {
  target += wiener_lpdf(y | a, t0, w, v);
}
