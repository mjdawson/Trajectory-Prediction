
HOMEANNOTATIONS = 'http://labelme2.csail.mit.edu/Release3.0/Annotations/users/Ben90';
HOMEIMAGES = 'http://labelme2.csail.mit.edu/Release3.0/Images/users/Ben90/';

D = LMdatabase(HOMEANNOTATIONS);

for i = 1:30
    nrows = str2num(D(i).annotation.imagesize.nrows);
    ncols = str2num(D(i).annotation.imagesize.ncols);
    filename = D(i).annotation.filename;
    filename = filename(1:end-3);
    c = 'csv';
    fname = strcat(filename,c);
    
    objMask = arrayfun(@(x) imdilate(poly2mask(double(x.polygon.x),double(x.polygon.y),nrows,ncols),strel('disk',10)), D(i).annotation.object,'UniformOutput',0);

    listObjName = {D(i).annotation.object(:).name};
    %listObj = unique(listObjName);
    listObj = {'road', 'grass', 'inaccessible', 'sidewalk'};
    classifier = cellfun(@(x) find(strcmp(x,listObj)),listObjName);
    objMask = cat(3,objMask{:});
    objMask = repmat( reshape(classifier,1,1,[]), [size(objMask,1), size(objMask,2), 1] ) .* objMask;

    [~, maxInd] = max( objMask , [], 3 );
    [~,~,Z] = meshgrid(1:size(objMask,2),1:size(objMask,1),1:size(objMask,3));
    firstObj = repmat( maxInd , [1,1,size(objMask,3)] );
    objMask( (objMask > 0) & (Z ~= firstObj) ) = 0;
    objArray = sum(objMask,3);
    
    
    [~,nearestVal] = find(objArray,1,'first');
    disp(nearestVal);
    
    for j=1:nrows
        for k=1:ncols
            
            if objArray(j,k) == 0
               objArray(j,k) = nearestVal;
            else
               nearestVal = objArray(j,k);
            end
        end
    end
    
    
    %imagesc(objArray,[0 4])
    
    
    [Y,X] = meshgrid(1:ncols, 1:nrows);
    X = ((X-1)/(nrows-1)*2)-1;
    Y = ((Y-1)/(ncols-1)*2)-1;

    pixel_Labels = [X(:),Y(:),objArray(:)];
    %pixel_Labels = single(pixel_Labels);
    pixel_Labels(2:2:end,:) = [];
      
    vals = unique(pixel_Labels(:,end));
    disp(vals);
    
    fprintf('Saving iteration: %d\n',i);
    csvwrite(fname, pixel_Labels);
    pixel_Labels(1:10,:)

end


