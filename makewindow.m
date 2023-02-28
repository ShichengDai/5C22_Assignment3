function window = makewindow(y, x, N, graph)

[h, w, c] = size(graph);


%make blocks to the input graph to make a new matrix
block_h = nan(h, N, c);
block_w = nan(N, w + 2 * N, c);

%size of the new matirx is (h + 2 * N) * (w + 2 * N)

matrix_new = [block_h, graph, block_h];
matrix_new = [block_w; matrix_new; block_w];

%segment the new matrix with window
%the new centre of the window is (x + N, y + N)
window = matrix_new(y + (N + 1) / 2 : y + (3 * N - 1) / 2, ...
         x + (N + 1) / 2 : x + (3 * N - 1) / 2, :);

