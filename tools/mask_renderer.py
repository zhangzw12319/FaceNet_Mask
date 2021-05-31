import os, sys, datetime
import numpy as np
import os.path as osp
# sys.path.append('../face3d/')
sys.path.append("/media/zhangzhizhong/wjm/code/insightface_baseline_torch/face3d")
print(sys.path)
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel
import cv2
import mxnet as mx
import random


class MaskRenderer:
    def __init__(self,model_dir, render_only=False):
        self.bfm = MorphabelModel(osp.join(model_dir, 'BFM.mat'))
        self.index_ind = self.bfm.kpt_ind
        uv_coords = face3d.morphable_model.load.load_uv_coords(osp.join(model_dir, 'BFM_UV.mat'))
        self.uv_size = (224,224)
        self.mask_stxr =  0.1
        self.mask_styr = 0.33
        self.mask_etxr = 0.9
        self.mask_etyr =  0.7
        self.tex_h , self.tex_w, self.tex_c = self.uv_size[1] , self.uv_size[0],3
        texcoord = np.zeros_like(uv_coords)
        texcoord[:, 0] = uv_coords[:, 0] * (self.tex_h - 1)
        texcoord[:, 1] = uv_coords[:, 1] * (self.tex_w - 1)
        texcoord[:, 1] = self.tex_w - texcoord[:, 1] - 1
        self.texcoord = np.hstack((texcoord, np.zeros((texcoord.shape[0], 1))))
        self.X_ind = self.bfm.kpt_ind
        if not render_only:
            from image_3d68 import Handler
            self.if3d68_handler = Handler(osp.join(model_dir, 'if1k3d68'), 0, 192, ctx_id=0)


    def transform(self, shape3D, R):
        s = 1.0
        shape3D[:2, :] = shape3D[:2, :]
        shape3D = s * np.dot(R, shape3D)
        return shape3D

    def preprocess(self, vertices, w, h):
        R1 = mesh.transform.angle2matrix([0, 180, 180])
        t = np.array([-w // 2, -h // 2, 0])
        vertices = vertices.T
        vertices += t
        vertices = self.transform(vertices.T, R1).T
        return vertices

    def project_to_2d(self,vertices,s,angles,t):
        transformed_vertices = self.bfm.transform(vertices, s, angles, t)
        projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection
        return projected_vertices[self.bfm.kpt_ind, :2]

    def params_to_vertices(self,params  , H , W):
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t  = params
        fitted_vertices = self.bfm.generate_vertices(fitted_sp, fitted_ep)
        transformed_vertices = self.bfm.transform(fitted_vertices, fitted_s, fitted_angles,
                                                  fitted_t)
        transformed_vertices = self.preprocess(transformed_vertices.T, W, H)
        image_vertices = mesh.transform.to_image(transformed_vertices, H, W)
        return image_vertices

    def build_params(self, face_image):
        if self.if3d68_handler.get(face_image) is None:
            return None

        landmark = self.if3d68_handler.get(face_image)[:,:2]
        #print(landmark.shape, landmark.dtype)
        if landmark is None:
            return None #face not found
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = self.bfm.fit(landmark, self.X_ind, max_iter = 3)
        return [fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t]

    def generate_mask_uv(self,mask, positions):
        uv_size = (self.uv_size[1], self.uv_size[0], 3)
        h, w, c = uv_size
        uv = np.zeros(shape=(self.uv_size[1],self.uv_size[0], 3), dtype=np.uint8)
        stxr, styr  = positions[0], positions[1]
        etxr, etyr = positions[2], positions[3]
        stx, sty = int(w * stxr), int(h * styr)
        etx, ety = int(w * etxr), int(h * etyr)
        height = ety - sty
        width = etx - stx
        mask = cv2.resize(mask, (width, height))
        uv[sty:ety, stx:etx] = mask
        return uv

    def render_mask(self,face_image, mask_image, params, auto_blend = True, positions=[0.1, 0.33, 0.9, 0.7]):
        uv_mask_image = self.generate_mask_uv(mask_image, positions)
        h,w,c = face_image.shape
        image_vertices = self.params_to_vertices(params ,h,w)
        output = (1-mesh.render.render_texture(image_vertices, self.bfm.full_triangles , uv_mask_image, self.texcoord, self.bfm.full_triangles, h , w ))*255
        output = output.astype(np.uint8)
        if auto_blend:
            mask_bd = (output==255).astype(np.uint8)
            final = face_image*mask_bd + (1-mask_bd)*output
            return final
        return output


if __name__ == "__main__":
    tool = MaskRenderer('/media/zhangzhizhong/wjm/code/insightface_baseline_torch/asset_mask')
    test_image_path = "/media/zhangzhizhong/wjm/code/insightface_baseline_torch/test_images/Tom_Hanks_54745.png"
    image = cv2.imread(test_image_path)
    print(type(image))
    print(image.shape)
    mask_image_path = "mask_images/mask7.png"
    # mask_image_path = "mask_images/mask1.jpg"
    # mask_image_path = "mask_images/mask3.png"
    #mask_image  = cv2.imread("masks/mask1.jpg")
    #mask_image  = cv2.imread("masks/black-mask.png")
    mask_image  = cv2.imread(mask_image_path)
    params = tool.build_params(image)

    # entire mask
    mask_out = tool.render_mask(image, mask_image, params)# use single thread to test the time cost

    # half mask
    # mask_out = tool.render_mask(image, mask_image, params, positions=[0.1, 0.5, 0.9, 0.7])

    print(type(mask_out))
    print(mask_out.shape)

    output_file = "mask_test_images"
    if not os.path.isdir(output_file):
        os.makedirs(output_file)
    image_name = test_image_path.split("/")[-1].split(".")[0] + "_" + mask_image_path.split("/")[-1].split(".")[0] + ".jpg"
    cv2.imwrite(os.path.join(output_file, image_name), mask_out)
    cv2.imwrite(os.path.join(output_file, "Tom_Hanks_54745.jpg"), image)

    # render masked ms1m
    read_record = mx.recordio.MXIndexedRecordIO("/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train.idx",
                                                "/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train.rec", "r")

    write_record_masked = mx.recordio.MXIndexedRecordIO("/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_masked2.idx",
                                                        "/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_masked2.rec", "w")

    write_record_all = mx.recordio.MXIndexedRecordIO("/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_all2.idx",
                                                     "/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_all2.rec2", "w")

    mask_images_list = []
    for path in os.listdir("mask_images"):
        mask_images_list.append(os.path.join("mask_images",path))
    print(mask_images_list)

    item1 = read_record.read_idx(0)
    h, s = mx.recordio.unpack(item1)
    # write_record_masked.write_idx(0, item1)
    # h_new = mx.recordio.IRHeader(flag=h.flag, label=h.label * 2.0, id=h.id, id2=h.id2)
    # write_record_all.write_idx(0, mx.recordio.pack(h_new, s))

    n_masked = 1
    n_all = 1
    for idx in read_record.keys:
        item = read_record.read_idx(idx)
        header, img_byte = mx.recordio.unpack(item)
        img = mx.image.imdecode(img_byte).asnumpy()
        # print(img.shape)
        # print(type(img))
        params = tool.build_params(img)
        if params is None:
            print(idx)
            continue

        mask_image = cv2.imread(mask_images_list[(n_masked-1) % len(mask_images_list)])
        img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]

        # entire mask
        mask_out = tool.render_mask(img, mask_image, params)
        # if idx == 1:
        #     cv2.imwrite(os.path.join(output_file, "test0.jpg"), mask_out)

        hh = mx.recordio.IRHeader(flag=header.flag, label=header.label, id=n_masked, id2=header.id2)
        s = mx.recordio.pack_img(hh, mask_out, quality=95, img_fmt=".jpg")
        write_record_masked.write_idx(n_masked, s)
        n_masked = n_masked + 1

        hh = mx.recordio.IRHeader(flag=header.flag, label=header.label, id=n_all, id2=header.id2)
        s = mx.recordio.pack_img(hh, img, quality=95, img_fmt=".jpg")
        write_record_all.write_idx(n_all, s)
        n_all = n_all + 1

        hh = mx.recordio.IRHeader(flag=header.flag, label=header.label, id=n_all, id2=header.id2)
        s = mx.recordio.pack_img(hh, mask_out, quality=95, img_fmt=".jpg")
        write_record_all.write_idx(n_all, s)
        n_all = n_all + 1

    print("all---------------")
    print(n_all)
    print("masked---------------")
    print(n_masked)

    h_masked_label = h.label
    h_masked_label = np.array(h_masked_label)
    h_masked_label[0] = float(n_masked)
    h_masked_label[1] = h.label[0]
    h_masked_label = np.asarray(h_masked_label)
    h_masked = mx.recordio.IRHeader(flag=h.flag, label=h_masked_label, id=h.id, id2=h.id2)
    print(h_masked)
    write_record_masked.write_idx(0, mx.recordio.pack(h_masked, s))

    h_all_label = h.label
    h_all_label = np.array(h_all_label)
    h_all_label[0] = float(n_all)
    h_all_label[1] = h.label[0] * 2.0
    h_all_label = np.asarray(h_all_label)
    h_all = mx.recordio.IRHeader(flag=h.flag, label=h_all_label, id=h.id, id2=h.id2)
    print(h_all)
    write_record_all.write_idx(0, mx.recordio.pack(h_all, s))

    write_record_masked.close()
    write_record_all.close()
    read_record.close()

    # item1 = read_record.read_idx(0)
    # h, s = mx.recordio.unpack(item1)
    # write_record_masked.write_idx(0, item1)
    # write_record_masked.write(item1)
    # h_new = mx.recordio.IRHeader(flag=h.flag, label=h.label * 2.0, id=h.id, id2=h.id2)
    # write_record_all.write_idx(0, mx.recordio.pack(h_new,s))
    #
    # print(h)
    # print(s.decode())
    # read_record.close()
    #
    #
    # write_record_masked.close()
    # write_record_all.close()
    #
    #
    read_record = mx.recordio.MXIndexedRecordIO("/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_masked2.idx",
                                                "/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_masked2.rec", "r")
    item2 = read_record.read_idx(0)
    h, s = mx.recordio.unpack(item2)
    print(h)
    print(s.decode())

    item2 = read_record.read_idx(5)
    header, img_byte = mx.recordio.unpack(item2)
    img_numpy = mx.image.imdecode(img_byte).asnumpy()
    img_numpy[:, :, [0, 1, 2]] = img_numpy[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(output_file, "test.jpg"), img_numpy)
    print(header)
    read_record.close()

    read_record = mx.recordio.MXIndexedRecordIO("/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_all2.idx",
                                                "/media/zhangzhizhong/wjm/dataset/ms1m-retinaface-t1/train_all2.rec", "r")

    item3 = read_record.read_idx(10)
    header, img_byte = mx.recordio.unpack(item3)
    img_numpy = mx.image.imdecode(img_byte).asnumpy()
    # img_numpy[:, :, [0, 1, 2]] = img_numpy[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(output_file, "test2.jpg"), img_numpy)
    print(header)

    item3 = read_record.read_idx(0)
    h, s = mx.recordio.unpack(item3)
    print(h)
    print(s.decode())
    read_record.close()

