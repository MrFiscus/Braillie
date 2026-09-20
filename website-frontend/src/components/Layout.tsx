import {Outlet } from "react-router-dom";

// This is filler we might add a consistent header later.
function Layout() {
    return (
        <main>
            <Outlet />
        </main>
    )
}

export default Layout;