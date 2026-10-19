use axum::Router;
use tokio::task::JoinHandle;

pub struct Server {
    pub url: String,
    task: JoinHandle<()>,
}

impl Server {
    pub fn start(router: Router) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("fixture listener");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}", listener.local_addr().expect("fixture address"));
        let task = tokio::spawn(async move {
            axum::Server::from_tcp(listener)
                .expect("fixture server")
                .serve(router.into_make_service())
                .await
                .expect("serve fixture");
        });
        Self { url, task }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
