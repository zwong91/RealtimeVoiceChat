if __name__ == "__main__":
    import os
    from huggingface_hub import hf_hub_download
    from RealtimeTTS import TextToAudioStream
    import stanza
    stanza.download(lang="zh-hans", processors={"pos": "gsdsimp_charlm"})

    # Coqui model download helper functions
    def create_directory(path: str) -> None:
        """
        Creates a directory at the specified path if it doesn't already exist.

        Args:
            path: The directory path to create.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def lasinya_models(models_root: str = "models", model_name: str = "Lasinya") -> None:
        """
        Ensures the Coqui XTTS Lasinya model files are present locally.

        Checks for required model files (config.json, vocab.json, etc.) within
        the specified directory structure. If any file is missing, it downloads
        it from the 'KoljaB/XTTS_Lasinya' Hugging Face Hub repository.

        Args:
            models_root: The root directory where models are stored.
            model_name: The specific name of the model subdirectory.
        """
        base = os.path.join(models_root, model_name)
        create_directory(base)
        files = ["config.json", "vocab.json", "speakers_xtts.pth", "model.pth"]
        for fn in files:
            local_file = os.path.join(base, fn)
            if not os.path.exists(local_file):
                # Not using logger here as it might not be configured yet during module import/init
                print(f"👄⏬ Downloading {fn} to {base}")
                hf_hub_download(
                    repo_id="KoljaB/XTTS_Lasinya",
                    filename=fn,
                    local_dir=base
                )

    lasinya_models()
    def dummy_generator():
        yield "我喜欢读书。天气很好。我们去公园吧。今天是星期五。早上好。这是我的朋友。请帮我。吃饭了吗？我在学中文。晚安。"

    def synthesize(engine, generator):
        stream = TextToAudioStream(engine)

        print("Starting to play stream")
        stream.feed(generator)
        filename = "synthesis_chinese_" + engine.engine_name

        # ❗ use these for chinese: minimum_sentence_length = 2, minimum_first_fragment_length = 2, tokenizer="stanza", language="zh", context_size=2
        stream.play(
            minimum_sentence_length=2,
            minimum_first_fragment_length=2,
            output_wavfile=f"{filename}.wav",
            on_sentence_synthesized=lambda sentence: print("Synthesized: " + sentence),
            tokenizer="stanza",
            language="zh",
            context_size=2,
        )

        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(stream.text())

        engine.shutdown()

    def get_engine(name):
        if name == "coqui":
            from RealtimeTTS import CoquiEngine
            # ❗ use these for chinese: voice="female_chinese", language = "zh"   # you can exchange voice with you own
            return CoquiEngine(
                specific_model="Lasinya",
                local_models_path="./models",
                voice="reference_audio.wav", # using a chinese cloning reference gives better quality
                language="zh",
                speed=1.1,
                #use_deepspeed=True,
                thread_count=6,
                stream_chunk_size=8,
                overlap_wav_len=1024,
                load_balancing=True,
                load_balancing_buffer_length=0.5,
                load_balancing_cut_off=0.1,
                add_sentence_filter=False,
            )
        else:
            from RealtimeTTS import SystemEngine

            # ❗ use these for chinese: voice = "Huihui"   # or specify a different locally installed chinese tts voice
            return SystemEngine(voice="Huihui")

    for engine_name in ["coqui"]:
        print("Starting engine: " + engine_name)
        engine = get_engine(engine_name)

        print("Synthesizing with engine: " + engine_name)
        synthesize(engine, dummy_generator())
