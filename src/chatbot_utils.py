import os

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)


def get_chapter_list(subject):

    if subject == "Biology":
        subject_name = subject.lower()
        # ``data`` can be in the project root or alongside this package.
        data_dirs = (
            os.path.join(parent_dir, "data"),
            os.path.join(os.path.dirname(parent_dir), "data"),
            os.path.join(working_dir, "data"),
        )
        chapter_dir_names = (f"class_12{subject_name}", f"class_12_{subject_name}")
        chapters_dir = next(
            (
                os.path.join(data_dir, dir_name)
                for data_dir in data_dirs
                for dir_name in chapter_dir_names
                if os.path.isdir(os.path.join(data_dir, dir_name))
            ),
            None,
        )
        if chapters_dir is None:
            raise FileNotFoundError(
                "Could not find the Biology chapters directory in: "
                + ", ".join(data_dirs)
            )
        chapters_list = os.listdir(chapters_dir)
        chapters_list = [x[:-4] for x in chapters_list]
        chapters_list.sort(key=lambda x: int(x.split(".")[0]))
        return chapters_list
