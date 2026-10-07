import argparse
import time
from pathlib import Path


import cv2
import numpy as np
import pandas as pd
import torch

from apr.config import ROOT
from apr.data.jaad import load_annotations
from apr.perception.backbone import AdapterBank

p_behave = ["action", "look", "cross", "occlusion"]
s_context = ["ped_density", "weather", "time_of_day", "road_presence"]

# map scene adpater to df column
s_cols = {
    "ped_density": "pedestrian_density", 
    "weather": "weather",
    "time_of_day": "time_of_day", 
    "road_presence": "road_presence"
}

def pad_box(b, w, h, padding=0.05):
    # added 5 percent padding from older script
    x1 = b[0]
    y1 = b[1]
    x2 = b[2]
    y2 = b[3]

    
    b_width = x2 - x1
    b_height = y2 - y1
    
    nx1 = int(max(0, x1 - padding * b_width))
    ny1 = int(max(0, y1 - padding * b_height))


    nx2 = int(min(w, x2 + padding * b_width))
    ny2 = int(min(h, y2 + padding * b_height))
    
    return (nx1, ny1, nx2, ny2)

def sync_gpu(dev):
    # need to wait for gpu otherwise time wouldbe completely wrong
    if "mps" in str(dev):
        torch.mps.synchronize()
    if "cuda" in str(dev):
        torch.cuda.synchronize()

def do_adapters(bank, img_tensors, adp_names):
    # nested with looks less weird than , 
    with torch.no_grad():
        with bank.model.disable_adapter():
            # get base features for router later
            feats = bank.model(pixel_values=img_tensors).logits
            
    p_dict = {}
    time_dict = {}
    
    for adp in adp_names:
        sync_gpu(bank.device)
        start_t = time.time()
        
        probs, _ = bank.predict(adp, img_tensors)
        
        sync_gpu(bank.device)
        end_t = time.time()
        
        p_dict[adp] = probs.cpu().numpy()
        time_dict[adp] = ((end_t - start_t) * 1000) / len(img_tensors)
        
    return feats.cpu().numpy().astype(np.float16), p_dict, time_dict

def save_results(dir_path, f_name, metad, arrs):
    dir_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = dir_path / str(f_name + "_meta.csv")
    pd.DataFrame(metad).to_csv(csv_path, index=False)
    
    np_dict = {}
    for k in arrs:
        np_dict[k] = np.concatenate(arrs[k])
        
    npz_path = dir_path / str(f_name + ".npz")
    np.savez_compressed(npz_path, **np_dict)
    print("saved", f_name, len(metad), "rows")

def main(split_name, stride_val, max_vid, out_dir):
    df = load_annotations()
    
    # filter df
    df = df[(df.split == split_name) & (df.frame_id % stride_val == 0)]
    
    all_vids = sorted(df.video_id.unique())
    if max_vid != None:
        all_vids = all_vids[:max_vid]
        
    my_bank = AdapterBank()

    ped_m = []
    scene_m = []
    
    # manually build dictionaries instead of list comps
    p_arrays = {}
    p_arrays['cls'] = []
    for n in p_behave:
        p_arrays["p_" + n] = []
        p_arrays["ms_" + n] = []
        
    s_arrays = {}
    s_arrays['cls'] = []
    for n in s_context:
        s_arrays["p_" + n] = []
        s_arrays["ms_" + n] = []

    count = 0
    for v_id in all_vids:
        # group all rows by frame so we dont loop whole df
        v_df = df[df.video_id == v_id]
        f_dict = {}
        for f_id, f_data in v_df.groupby("frame_id"):
            f_dict[f_id] = f_data
            
        vid_path = str(ROOT) + "/JAAD/JAAD_clips/video_" + str(v_id) + ".mp4"
        cap = cv2.VideoCapture(vid_path)
        
        f_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if f_num in f_dict:
                curr_rows = f_dict[f_num]
                
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h = img_rgb.shape[0]
                w = img_rgb.shape[1]

                # do scene stuff first
                sc_input = my_bank.preprocess([img_rgb])
                cls_out, p_out, t_out = do_adapters(my_bank, sc_input, s_context)
                
                row_one = curr_rows.iloc[0]
                
                s_meta_row = {
                    "video_id": v_id, 
                    "frame_id": f_num, 
                    "split": split_name, 
                    "num_peds": len(curr_rows)
                }
                for sc in s_context:
                    s_meta_row[sc] = row_one[s_cols[sc]]
                    
                scene_m.append(s_meta_row)
                s_arrays["cls"].append(cls_out)
                
                for sc in s_context:
                    s_arrays["p_" + sc].append(p_out[sc])
                    s_arrays["ms_" + sc].append([t_out[sc]])

                # crop out peds
                crop_list = []
                valid_rows = []
                
                for idx, row in curr_rows.iterrows():
                    box = (row.bbox_xtl, row.bbox_ytl, row.bbox_xbr, row.bbox_ybr)
                    nx1, ny1, nx2, ny2 = pad_box(box, w, h)
                    
                    # check if valid box
                    if nx2 > nx1 and ny2 > ny1:
                        crop_img = img_rgb[ny1:ny2, nx1:nx2]
                        crop_list.append(crop_img)
                        valid_rows.append(row)
                        
                if len(crop_list) > 0:
                    p_input = my_bank.preprocess(crop_list)
                    p_cls, p_probs, p_times = do_adapters(my_bank, p_input, p_behave)
                    
                    for r in valid_rows:
                        p_row = {
                            "video_id": v_id, 
                            "frame_id": f_num, 
                            "split": split_name,
                            "track_id": r.track_id, 
                            "label": r.label, 
                            "num_peds": len(valid_rows),
                            "x1": r.bbox_xtl, "y1": r.bbox_ytl, 
                            "x2": r.bbox_xbr, "y2": r.bbox_ybr
                        }
                        for pb in p_behave:
                            p_row[pb] = r[pb]
                        ped_m.append(p_row)
                        
                    p_arrays["cls"].append(p_cls)
                    for pb in p_behave:
                        p_arrays["p_" + pb].append(p_probs[pb])
                        # duplicate time array for each ped
                        p_arrays["ms_" + pb].append([p_times[pb]] * len(valid_rows))
                        
            f_num += 1
        
        count += 1
        print("done with [" + str(count) + "/" + str(len(all_vids)) + "] video_" + str(v_id))

    save_results(out_dir, "jaad_" + split_name + "_ped", ped_m, p_arrays)
    save_results(out_dir, "jaad_" + split_name + "_scene", scene_m, s_arrays)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-videos", type=int, default=None)
    
    # cast to str just in case path object messes up default
    p.add_argument("--out", default=str(ROOT / "Generated_Data" / "cache"))
    
    args = p.parse_args()
    out_folder = Path(args.out)
    
    main(args.split, args.stride, args.max_videos, out_folder)
