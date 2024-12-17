from preprocess.tools import list_wav_files,  generate_MFCCs


if __name__ == "__main__":
    root_dir = "speech_commands_v0.02"
    exclude_folders = ["_background_noise_"]

    #print(list_files(root_dir, exclude_folders))

    files = list_wav_files(root_dir, exclude_folders)
    #print(find_max_samples_wav(files))

    generate_MFCCs(files, "dataset", 20, 16000, 2048, 512)



