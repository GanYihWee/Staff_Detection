import math


class EE_Detection:
    def __init__(self, thres):
            
        self.threshold = thres
        

    def draw_bounding_box(self, boxes, shape):

        coords_arr = []
        if len(boxes) >1:
            for val in boxes:
                # only class 0 (person) and those above threshold
                if val[4].cpu().numpy() > self.threshold and val[5].cpu().numpy() == 0:
                    x1 = int(val[0].cpu().numpy()) if int(val[0].cpu().numpy()) > 0 else 1
                    y1 = int(val[1].cpu().numpy()) if int(val[1].cpu().numpy()) > 0 else 1
                    x2 = math.floor(val[2].cpu().numpy()) if math.floor(val[2].cpu().numpy()) < shape[1] else shape[1]
                    y2 = math.floor(val[3].cpu().numpy()) if math.floor(val[3].cpu().numpy()) < shape[0] else shape[0]
                    coords_arr.append([x1,y1,x2,y2])


        return coords_arr if len(coords_arr) > 0 else False   


    def detect(self, results, shape):
    
        output = self.draw_bounding_box(results.xyxy[0], shape)
        
        if output == False:
            return output
        
        elif len(output) >0:
            return output
