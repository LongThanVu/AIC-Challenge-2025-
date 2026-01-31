#[derive(Debug, Clone, diesel::Queryable, diesel::Selectable)]
#[diesel(table_name = crate::schema::videos)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct VideoEntity {
    id: i64,
    name: String,
    watch_url: String,
}

impl VideoEntity {
    pub fn id(&self) -> i64 {
        self.id
    }

    pub fn watch_url_mut(&mut self) -> &mut String {
        &mut self.watch_url
    }
    
    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }
}
