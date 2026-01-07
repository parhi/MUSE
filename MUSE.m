function [selected_fea, J_history, elim_frac, survivors_mask] = ...
    MUSE(X, y, m, K, p, T, Ts)
% Minimum Uncertainty and Sample Elimination (MUSE) feature selection.
%
% This function implements the core MUSE algorithm for binary classification:
%   1) Discretize each (continuous) feature into K bins using equal-probability
%      (quantile) binning.
%   2) For each candidate feature Xi, compute an "uncertainty" score:
%        - For each bin Bk of Xi, compute conditional entropy H(c | Bk)
%          where c is the class label.
%        - Sort bins by increasing H(c | Bk) (best/most-certain bins first).
%        - Take the smallest number of bins K0 whose cumulative probability mass
%          exceeds p (i.e., sum_{k=1..K0} P(Bk) > p).
%        - Score: J(Xi) = sum_{k=1..K0} P(Bk) * H(c | Bk).
%      Lower J(Xi) is better.
%   3) Select the feature with minimum J(Xi).
%   4) Sample elimination: for the selected feature, compute each bin's impurity
%        I(Bk) = min(P(class=0 | Bk), P(class=1 | Bk)).
%      Eliminate (discard) all samples that fall into "good" bins where I(Bk) < T.
%   5) Repeat steps 1–4 on the remaining (surviving) samples until you select m
%      features or the stopping criterion triggers.
%
% INPUTS
%   X   : [N x D] matrix of features (N samples, D features).
%         Features are assumed continuous (or at least not pre-discretized).
%   y   : [N x 1] binary labels. Accepts {0,1} or {-1,+1} or any two numeric values.
%   m   : maximum number of features to select.
%   K   : number of bins for equal-probability discretization (e.g., 10–30).
%   p   : fraction of samples used in the uncertainty score (e.g., 0.2).
%         Only the "most certain" bins covering > p mass contribute to J(Xi).
%   T   : impurity threshold for elimination (e.g., 0.1).
%         Bins with impurity < T are considered "good" and their samples are removed.
%   Ts  : stopping threshold for class survival fraction (e.g., 0.01 or 0.1).
%         If the fraction of surviving samples of either class drops below Ts,
%         the algorithm stops early. Set Ts = 0 to disable.
%
% OUTPUTS
%   selected_fea    : [1 x k] indices of selected features (k <= m).
%   J_history       : [1 x k] uncertainty scores for the selected features.
%   elim_frac       : [1 x k] fraction of samples eliminated at each iteration.
%   survivors_mask  : [N x (k+1)] logical matrix tracking surviving samples.
%                     Column 1 is the initial all-true mask. Column (t+1) is the
%                     mask after iteration t.
%
% NOTE
%   - MUSE is designed for binary classification problem only.
%   - For EEG/PSD-like features, K around 10–30 usually works better than very
%     large K (which creates sparse bins and unstable estimates).
%
% Example:
%   [idx, J, elim, surv] = MUSE(X, y, 10, 20, 0.2, 0.1, 0.01);

    %-----------------------------
    % 0) Basic checks / formatting
    %-----------------------------
    [N, D] = size(X);
    y = y(:);

    if numel(y) ~= N
        error('y must have the same number of rows as X.');
    end

    % Normalize labels to 0/1 (logical or numeric)
    uy = unique(y);
    if numel(uy) ~= 2
        error('y must be binary (exactly two unique label values).');
    end
    % Map larger label -> 1, smaller -> 0 (works for {0,1}, {-1,1}, etc.)
    y01 = (y == max(uy));

    % Initial class counts (used by the Ts stopping rule)
    N0_init = sum(~y01);
    N1_init = sum( y01);

    % Track which features have already been selected
    selected = false(1, D);

    % Preallocate maximum output sizes; 
    selected_fea = zeros(1, m);
    J_history   = nan(1, m);
    elim_frac   = nan(1, m);

    % Survivors mask: samples that remain "hard" and continue to next iteration
    survivors = true(N, 1);
    survivors_mask = false(N, m+1);
    survivors_mask(:,1) = survivors;

    iter = 0;

    %-----------------------------
    % Main MUSE loop (select 1 feature per iteration)
    %-----------------------------
    while iter < m
        iter = iter + 1;

        % Work only on currently surviving samples
        idx_surv = find(survivors);
        Xd = X(idx_surv, :);
        yd = y01(idx_surv);
        Ns = numel(yd);

        % If only one class remains (or no samples), nothing more to do
        if Ns == 0 || numel(unique(yd)) < 2
            iter = iter - 1;
            break;
        end

        %-----------------------------
        % 1) Optional stopping criterion based on surviving class fractions
        %-----------------------------
        if Ts > 0
            frac0 = sum(~yd) / max(N0_init,1);
            frac1 = sum( yd) / max(N1_init,1);
            if (frac0 < Ts) || (frac1 < Ts)
                iter = iter - 1;
                break;
            end
        end

        %-----------------------------
        % 2) Score each candidate feature using MUSE uncertainty J(Xi)
        %-----------------------------
        J = inf(1, D);           % uncertainty scores for all features
        bin_info = cell(1, D);   % store per-feature info needed for elimination

        for f = 1:D
            if selected(f)
                continue; % skip already-chosen features
            end

            % 2a) Discretize feature f into K equal-probability bins on surviving samples
            x = Xd(:, f);
            b = equalprob_bins(x, K);  % b is an integer bin index per sample

            % 2b) Compute per-bin: probability P(Bk), conditional entropy H(c|Bk),
            %     and impurity I(Bk) = min(P0, P1)
            bins = unique(b);
            nb = numel(bins);

            Pk = zeros(nb,1);
            Hk = zeros(nb,1);
            impurity = zeros(nb,1);

            for k = 1:nb
                in_bin = (b == bins(k));
                nk = sum(in_bin);
                if nk == 0
                    continue;
                end

                Pk(k) = nk / Ns;

                % Class proportions inside bin
                p1 = sum(yd(in_bin)) / nk; % P(class=1 | bin)
                p0 = 1 - p1;               % P(class=0 | bin)

                % Conditional entropy H(c | Bk) in bits
                % Pure bins (p1=0 or 1) have H=0.
                if p1 > 0 && p1 < 1
                    Hk(k) = -p1*log2(p1) - p0*log2(p0);
                else
                    Hk(k) = 0;
                end

                % Impurity used for elimination
                impurity(k) = min(p0, p1);
            end

            % 2c) Sort bins by increasing entropy (most certain bins first)
            [Hk_sorted, ord] = sort(Hk, 'ascend');
            Pk_sorted = Pk(ord);

            % 2d) Find smallest K0 such that cumulative mass exceeds p
            cumP = cumsum(Pk_sorted);
            K0 = find(cumP > p, 1, 'first');
            if isempty(K0)
                K0 = nb;
            end

            % 2e) Uncertainty score J(Xf)
            J(f) = sum(Pk_sorted(1:K0) .* Hk_sorted(1:K0));

            % Store info for elimination if this becomes the selected feature
            bin_info{f} = struct('bins', bins, 'b', b, 'impurity', impurity);
        end

        %-----------------------------
        % 3) Select feature with minimum uncertainty score
        %-----------------------------
        [Jmin, fstar] = min(J);
        if ~isfinite(Jmin)
            iter = iter - 1;
            break;
        end

        selected(fstar) = true;
        selected_fea(iter) = fstar;
        J_history(iter) = Jmin;

        %-----------------------------
        % 4) Sample elimination using impurity threshold T
        %    Remove all samples that fall in "good" bins where impurity < T
        %-----------------------------
        info = bin_info{fstar};
        remove_local = false(Ns, 1); % in surviving-sample coordinates

        for k = 1:numel(info.bins)
            if info.impurity(k) < T
                remove_local = remove_local | (info.b == info.bins(k));
            end
        end

        % Update global survivors mask
        before = sum(survivors);
        survivors(idx_surv(remove_local)) = false;
        after = sum(survivors);

        elim_frac(iter) = (before - after) / max(before,1);
        survivors_mask(:, iter+1) = survivors;
    end

    %-----------------------------
    % Trim preallocated outputs to actual number of iterations
    %-----------------------------
    selected_fea   = selected_fea(1:iter);
    J_history      = J_history(1:iter);
    elim_frac      = elim_frac(1:iter);
    survivors_mask = survivors_mask(:, 1:iter+1);
end


function b = equalprob_bins(x, K)
%
%EQUALPROB_BINS  Equal-probability (quantile) discretization into K bins.
%
% Given a continuous vector x, this function returns an integer vector b
% of the same length, where b(i) indicates the bin index assigned to x(i).
%
% "Equal-probability" means bin edges are set by quantiles so that each bin
% contains roughly N/K samples (ties can reduce the number of distinct bins).
%


    x = x(:);

    % Constant feature or trivial bin count -> one bin
    if K <= 1 || all(x == x(1))
        b = ones(size(x));
        return;
    end

    % Quantile edges: K bins -> K+1 edges
    edges = quantile(x, linspace(0, 1, K+1));

    % Remove duplicate edges created by ties/repeated values
    edges = unique(edges, 'stable');
    if numel(edges) < 2
        b = ones(size(x));
        return;
    end

    % Assign bins
    b = discretize(x, edges);

    % Numerical safety: discretize can produce NaNs at boundaries
    if any(isnan(b))
        % Clamp x to [min,max] and re-discretize
        xmin = edges(1); xmax = edges(end);
        x2 = min(max(x, xmin), xmax);
        b = discretize(x2, edges);
        b(isnan(b)) = 1;
    end
end