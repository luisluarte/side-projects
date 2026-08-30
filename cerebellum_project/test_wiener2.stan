data {
  array[10] real y;
}
parameters {
  vector[10] a;
  real t0;
  real w;
  vector[10] v;
}
model {
  target += wiener_lpdf(y | a, t0, w, v);
}
