function nombrpregunta3 ()
  clc; clear;

%  Datos generales
  sizeLenght = 40;
  numImages = 9;
  totalImg = 360;
  totalPixels = 10304;


  % Paso 1

  S = zeros(totalPixels, totalImg);
  columna_s = 1;
  for k = 1:sizeLenght
    for m = 1:numImages
      directory=['training/s',num2str(k),'/',num2str(m),'.jpg'];
      A=imread(directory);
      B=im2double(A);
      x = B(:);
      S(:, columna_s) = x;
      columna_s = columna_s + 1;
    end
  end

  % Paso 2
  fMean = zeros(totalPixels, 1);
  % Calcular la suma de cada fila
  for i = 1:totalPixels
      fMean(i) = sum(S(i, :));
  endfor
  fMean = 1/totalImg * fMean;

  % Paso 3 y 4
  A = zeros(totalPixels, totalImg);
  for i = 1:totalImg
      A(:,i) = S(:,i) - fMean;
  endfor

  % Paso 5
  [U, S, V] = svdCompact(A);

  % Paso 13


  for n = 1: sizeLenght;

    selectedImg = '';
    minValue = 0;
    direccion=['compare/p',num2str(n),'.jpg'];
    A1=imread(direccion);
    B=im2double(A1);
    f = B(:);
    x = U'*(f-fMean);

    for k = 1:sizeLenght
      for m = 1:numImages
        direccion=['training/s',num2str(k),'/',num2str(m),'.jpg'];
        A=imread(direccion);
        B=im2double(A);
        f = B(:);
        xi = U'*(f-fMean);
        dif = ((x-xi)'*(x-xi))^(1/2);
        if minValue == 0;
          minValue = dif;
        elseif dif < minValue;
          minValue = dif;
          selectedImg = direccion;
        end
      end
    end

    subplot(1,2,1);
    imshow(A1) %Mostrar imagen
    title('Imagen Buscada')

    A2 = imread(selectedImg);
    subplot(1,2,2);
    imshow(A2) %Mostrar imagen
    title('Imagen Encontrada')
    pause(0.2)

  endfor

endfunction







