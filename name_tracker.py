from collections import OrderedDict, Counter


class NamesTracker:
    def __init__(self, maxDisappeared=10, maxNumFrame=24, maxAppeared=20):
        self.nextObjectID = 0
        self.objects = OrderedDict()  # Tên đối tượng
        self.disappeared = OrderedDict()  # Đếm số frame hình đối tượng chưa xuất hiện
        self.appeared = OrderedDict()  # Đếm số frame hình đối tượng đã xuất hiện
        self.numFrame = OrderedDict()  # Đếm tổng số frame hình
        self.flag = OrderedDict()  # Cờ theo dỏi đối tượng đã được cập nhật hay chưa
        self.emotions = OrderedDict()  # List chứa cảm xúc của đồi tượng qua các frame hình

        self.maxDisappeared = maxDisappeared
        self.maxNumFrame = maxNumFrame
        self.maxAppeared = maxAppeared

    # Hàm đăng ký đối tượng
    def register(self, names, emotion):
        self.objects[self.nextObjectID] = names
        self.emotions[self.nextObjectID] = [emotion]
        self.appeared[self.nextObjectID] = 1
        self.disappeared[self.nextObjectID] = 0
        self.numFrame[self.nextObjectID] = 1
        self.flag[self.nextObjectID] = False
        self.nextObjectID += 1

    # Hàm hủy đối tượng
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.appeared[objectID]
        del self.disappeared[objectID]
        del self.numFrame[objectID]
        del self.flag[objectID]
        del self.emotions[objectID]

    # Hàm cập nhật đối tượng khi không phát hiện được mặt
    def update_no_faces(self):
        for objectID in list(self.disappeared.keys()):
            self.disappeared[objectID] += 1
            self.numFrame[objectID] += 1
            self.flag[objectID] = True
            # self.appeared[objectID] = 0
            if (self.disappeared[objectID] > self.maxDisappeared) or (self.numFrame[objectID] > self.maxNumFrame):
                self.deregister(objectID)
        # return self.objects

    # Hàm cập nhật đối tượng khi phát hiện có mặt
    def update(self, name, emotion):
        if len(self.objects) == 0:  # Không có đối tượng nào đang theo dõi
            self.register(name, emotion)  # tiến hành thêm đối tượng mới
        else:
            # Có đối tượng đang được theo dõi
            for objectID in list(self.flag.keys()):  # Gán cờ  cập nhật của tất cả đối tượng đang theo dõi là False
                self.flag[objectID] = False
            # Tiến hành cập nhật các đối tượng đang theo dõi
            self.update_appeared(name, emotion)
            # Cập nhật lại những đối tượng chưa đưa xét, mỗi đối tượng chỉ cập nhật một lần
            self.update_disappeared(name)
        for objectID in list(self.disappeared.keys()):
            if (self.disappeared[objectID] > self.maxDisappeared) or (self.numFrame[objectID] > self.maxNumFrame):
                self.deregister(objectID)

    # Hàm điểm danh
    def attendance(self):
        namesObject = []
        emotionObject = []
        for objectID in list(self.appeared.keys()):
            if self.appeared[objectID] > self.maxAppeared:
                namesObject.append(self.objects[objectID])
                c = Counter(self.emotions[objectID])
                mode = c.most_common(1)
                emotionObject.append(mode[0][0])
                self.deregister(objectID)
        return namesObject, emotionObject

    # Hàm cập nhật đối tượng chưa xuật hiện
    def update_disappeared(self, name):
        for objectID in list(self.objects.keys()):
            if self.objects[objectID] != name:
                if not self.flag[objectID]:
                    self.disappeared[objectID] += 1
                    self.numFrame[objectID] += 1
                    self.flag[objectID] = True

    # Hàm cập nhật đối tượng đã xuất hiện
    def update_appeared(self, name, emotion):
        for objectID in list(self.objects.keys()):
            # Nếu name chuyền vào trùng với name đang theo dõi tiến hành cập nhật đối tượng, gán cờ cập nhật của
            # đối tượng là True và thoát
            if self.objects[objectID] == name:
                self.appeared[objectID] += 1
                # self.disappeared[objectID] = 0
                self.numFrame[objectID] += 1
                self.emotions[objectID].append(emotion)
                self.flag[objectID] = True
                break
        else:
            # Nếu name chuyền vào không có trong danh sách đang theo dõi tiến hành thêm đối tượng mới
            self.register(name, emotion)