clear all;
HOMEIMAGES = 'PATH TO YOUR IMAGES';
HOMEANNOTATIONS = 'PATH TO YOUR ANNOTATIONS';
output_folder='PATH TO YOUR OUTPUT FOLDER';
ext='.tif';


D = LMdatabase(HOMEANNOTATIONS);
[Dwire, j] = LMquery(D, 'object.name', 'wire');


for i=1:4
    
image_path=strcat(HOMEIMAGES,'users/_/fyp/',Dwire(i).annotation.filename);
[path,filename,~] = fileparts(image_path);
new_image_path=strcat(output_folder,filename,ext);
mask_path=strcat(output_folder,filename,'_mask',ext);

image=imread(image_path);
[mask, class] = LMobjectmask(Dwire(i).annotation, HOMEIMAGES);

if size(mask,3)~=1
    a=mask(:,:,1);
    b=mask(:,:,2);
    mask=a+b;
end
imwrite(image,new_image_path);
imwrite(mask,mask_path);


end
