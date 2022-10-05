class CosmoDataset(Dataset):
    """CosmoAI  dataset."""

    def __init__(self, csvfile, imgdir):
        """
        Args:
            csvfile (string): Path to the csv file with annotations.
            imgdir (string): Directory with all the images.
        """
        self.frame = pd.read_csv(csv_file)
        self.imgdir = imgdir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.imgdir,
                                self.frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        return sample
