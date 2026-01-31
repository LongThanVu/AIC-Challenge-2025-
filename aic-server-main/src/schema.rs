// @generated automatically by Diesel CLI.

diesel::table! {
    keyframes (id) {
        id -> Int8,
        name -> Text,
        video_id -> Int8,
        frame_index -> Int8,
        frame_timestamp -> Float4,
    }
}

diesel::table! {
    videos (id) {
        id -> Int8,
        name -> Text,
        watch_url -> Text,
    }
}

diesel::joinable!(keyframes -> videos (video_id));

diesel::allow_tables_to_appear_in_same_query!(keyframes, videos,);
