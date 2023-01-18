from pathlib import Path
import meshio
import ref

ref_key = "stereobj_1m"
data_ref = ref.__dict__[ref_key]

model_dir = data_ref.model_dir
objects = data_ref.objects

original_files = list(Path("datasets/BOP_DATASETS/stereobj_1m/objects").glob("*.obj"))
print('original files', original_files)

for obj_file in original_files:
    obj_id = data_ref.obj2id[obj_file.stem]
    mesh = meshio.read(obj_file)
    # write as ascii file
    out_path = Path(model_dir) / "obj_{:06d}.ply".format(obj_id)
    print('write to ', str(out_path))
    meshio.write(out_path, mesh, file_format="ply", binary=False)


