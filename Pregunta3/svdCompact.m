% La funcion svdCompact realiza una descomposición compacta de valores singulares (SVD) de una matriz A de una manera mas optimizada.
% La SVD es una factorización matricial que descompone A en U Sigma V^T,
% donde U y V son matrices ortogonales y Sigma es una matriz diagonal con los valores singulares.
%
% Entradas:
%   A: La matriz que se va a descomponer.
%
% Salidas:
%   Ur: La matriz U reducida, que contiene únicamente las columnas correspondientes a los valores singulares no nulos.
%   Sr: La matriz diagonal reducida Sigma, que contiene únicamente los valores singulares no nulos.
%   Vr: La matriz V reducida, que contiene únicamente las columnas correspondientes a los valores singulares no nulos.
%
function [Ur, Sr, Vr] = svdCompact (A)
    [m,n] = size(A);
    if m > n
        M1 = A' * A;
        [V1, D] = eig(M1);
        y1 = diag(D);
        const = n * max(y1) * eps;
        y2 = (y1 > const);
        rA = sum(y2);
        y3 = y1.*y2; %rango de la matriz
        [s1, order] = sort(sqrt(y3), 'descend');
        V2 = V1(:, order);
        Vr = V2(:, 1: rA);
        Sr = diag(s1(1: rA));
        Ur = (1./(s1(1:rA))').*(A * Vr);
    else
        M1 = A * A';
        [U1, D] = eig(M1);
        y1 = diag(D);
        const = m * max(y1) * eps;
        y2 = (y1 > const);
        rA = sum(y2); %rango de la matriz
        y3 = y1.*y2;
        [s1, order] = sort(sqrt(y3), 'descend');
        U2 = U1(:, order);
        Ur = U2(:, 1: rA);
        Sr = diag(s1(1 : rA));
        Vr = (1./(s1(1 : rA))').*(A'*Ur);
    end
end
