#[derive(Debug, Clone, diesel::Queryable, diesel::Selectable)]
#[diesel(table_name = crate::schema::keyframes)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct KeyframeEntity {
    id: i64,
    name: String,
    frame_index: i64,
    frame_timestamp: f32,
}

impl KeyframeEntity {
    pub fn id(&self) -> i64 {
        self.id
    }

    pub fn frame_index(&self) -> i64 {
        self.frame_index
    }

    pub fn frame_timestamp(&self) -> f32 {
        self.frame_timestamp
    }

    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }
}
