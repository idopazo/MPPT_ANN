load '.\Datos_Medida\shadowandframe5000.mat'
for i=1:size(cellshadow,2)
     aux=min(cellshadow{1,i});
     minimos(i)=min(aux);
     clear aux
end